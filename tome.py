from typing import Type, IO
from subprocess import run
import os

from dataclasses import dataclass
from enum import IntEnum, auto

import argparse


# ---------------------------------------------------------------------------------------------------------------------
# CLI Implementation
#

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI for the Tome programming language.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--compile",
        action="store_true",
        help="Compile the file (targeting x86_64 linux)")
    group.add_argument(
        "-i", "--interpret",
        action="store_true",
        help="Interpret the file")

    parser.add_argument("filename", help="The path to the source file")

    args = parser.parse_args()
    ir = frontend(args.filename)

    if args.compile:
        do_compile(ir)
    else:
        do_interpret(ir)


def frontend(path: str) -> 'list[Instr]':
    with open(path, "r") as source_file:
        source = source_file.read()
    tokens = lex(source, path)
    instrs = parse(tokens)
    return instrs


def do_compile(instructions: 'list[Instr]') -> None:
    with open("out.asm", "w") as out:
        gen_code(instructions, Linux_x86_64, out)

    run(["nasm", "-felf64", "out.asm", "-o", "out.o"])
    run(["ld", "out.o", "-o", "out"])


def do_interpret(instructions: 'list[Instr]') -> None:
    interpret(instructions)


# ---------------------------------------------------------------------------------------------------------------------
# Lexer Implementation
#

class TokenType(IntEnum):
    KEYWORD = auto()
    NUMBER = auto()
    STRING = auto()
    WORD = auto()
    PUNC = auto()
    EOF = auto()


@dataclass
class Loc:
    file: str
    row: int
    col: int

    def shift(self, d: int) -> 'Loc':
        return Loc(self.file, self.row, self.col + d)

    def __repr__(self) -> str:
        return f"[{self.file} @ {self.row + 1}:{self.col + 1}]"


@dataclass
class Token:
    typ: TokenType
    lexeme: str
    loc: Loc


KEYWORDS = {"data", "inline", "def", "is", "if", "then", "else-if", "else", "do", "while", "->", "return"}
SPACE_CHARACTERS = {" ", "\n", "\t"}
PUNCTUATION_CHARACTERS = {";", ",", "[", "]"}


FILES_INCLUDED: list[str] = []


def lex(source: str, file_name: str) -> list[Token]:
    FILES_INCLUDED.append(file_name)

    pos = 0
    line_count, line_start = 0, 0

    tokens = []
    while pos < len(source):
        char = source[pos]

        # The location is the row (aka line) and the
        # column (how far we are past the start of the line)
        loc = Loc(file_name, line_count, pos - line_start)

        # Skip spaces, but be makes sure to track our position
        # in regard to the position in the line for reporting
        if char in SPACE_CHARACTERS:
            pos, lines_passed, new_start = lex_spaces(source, pos)
            if lines_passed > 0:
                line_count = line_count + lines_passed
                line_start = new_start

        elif char == "-" and pos + 1 < len(source) and source[pos + 1] == "-":
            pos = lex_comment(source, pos)

        # TODO: Maybe one day we will need punctuation >1 chars long
        elif char in PUNCTUATION_CHARACTERS:
            pos = pos + 1
            tokens.append(Token(TokenType.PUNC, char, loc))

        elif char.isdigit():
            pos, lexeme = lex_number(source, pos)

            # Could this be a word that starts with a number? If so, continue parsing
            if pos < len(source) and source[pos] not in SPACE_CHARACTERS | PUNCTUATION_CHARACTERS:
                pos, lexeme2 = lex_word(source, pos)
                tokens.append(Token(TokenType.WORD, f"{lexeme}{lexeme2}", loc))

            # Else it's definitely a number so return a number
            else:
                tokens.append(Token(TokenType.NUMBER, lexeme, loc))

        elif char == "\'":
            pos, lexeme = lex_char(source, pos, loc)
            tokens.append(Token(TokenType.NUMBER, lexeme, loc))

        elif char == "\"":
            pos, lexeme = lex_string(source, pos, loc)
            tokens.append(Token(TokenType.STRING, lexeme, loc))

        elif char == "$":
            # Preprocessor macro starts
            pos = pos + 1
            pos, directive = lex_word(source, pos)

            pos, lines_passed, new_start, tks = handle_preprocessor_directive(source, pos, directive, loc)
            if lines_passed > 0:
                line_count = line_count + lines_passed
                line_start = new_start

            tokens.extend(tks)

        # Anything that isn't specifically special should be treated as a plain
        # word, unless it matches a keyword from the list
        else:
            pos, lexeme = lex_word(source, pos)
            typ = TokenType.KEYWORD if lexeme in KEYWORDS else TokenType.WORD
            tokens.append(Token(typ, lexeme, loc))

    tokens.append(Token(TokenType.EOF, "EoF", Loc(file_name, line_count, pos - line_start)))
    return tokens


def lex_spaces(source: str, start: int) -> tuple[int, int, int]:
    pos = start
    lines_seen, line_start = 0, 0
    while pos < len(source) and source[pos] in SPACE_CHARACTERS:
        if source[pos] == "\n":
            lines_seen = lines_seen + 1
            line_start = pos + 1
        pos = pos + 1
    return pos, lines_seen, line_start


def lex_comment(source: str, start: int) -> int:
    pos = start
    while pos < len(source) and source[pos] != "\n":
        pos = pos + 1
    return pos


def lex_number(source: str, start: int) -> tuple[int, str]:
    pos = start
    while pos < len(source) and source[pos].isdigit():
        pos = pos + 1
    return pos, source[start:pos]


def lex_char(source: str, start: int, loc: Loc) -> tuple[int, str]:
    if source[start] != "\'":
        print(f"{loc} Error: Character literal must begin with a quote.")
        exit(1)

    pos = start + 1
    if not pos < len(source):
        print(f"{loc} Error: Unterminated Character literal.")
        exit(1)

    cur = source[pos]
    if cur == "\\":
        if not pos + 1 < len(source):
            print(f"{loc} Error: Unterminated Character literal.")
            exit(1)

        pos = pos + 1
        nxt = source[pos]
        if nxt == "n":
            lexeme = str(ord("\n"))
        elif nxt == "t":
            lexeme = str(ord("\t"))
        elif nxt == "0":
            lexeme = str(ord("\0"))
        elif nxt == "\'":
            lexeme = str(ord("\'"))
        else:
            print(f"{loc.shift(2)} Error: Unrecognised Escape Sequence.")
            exit(1)

    else:
        lexeme = str(ord(cur))

    pos = pos + 1
    if not (pos < len(source) and source[pos] == "\'"):
        print(f"{loc} Error: Unterminated Character literal.")
        exit(1)

    pos = pos + 1
    return pos, lexeme


def lex_string(source: str, start: int, loc: Loc) -> tuple[int, str]:
    if source[start] != "\"":
        print(f"{loc} Error: String must begin with a quote.")
        exit(1)

    string_content = ""

    pos = start + 1
    while pos < len(source):
        char = source[pos]

        if char == "\"":
            pos = pos + 1
            return pos, string_content

        elif char == "\\":
            if not pos + 1 < len(source):
                print(f"{loc} Error: Unterminated string.")
                exit(1)

            next_ = source[pos + 1]
            pos = pos + 2
            if next_ == "\\" or next_ == "\"":
                string_content += next_
            elif next_ == "n":
                string_content += "\n"
            elif next_ == "t":
                string_content += "\t"
            elif next_ == "0":
                string_content += "\0"
            else:
                print(f"{loc.shift(pos - start + 1)} Error: Unrecognised escape sequence.")
                exit(1)

        elif char == "\n":
            print(f"{loc} Error: Unterminated string.")
            exit(1)

        else:
            string_content += char
            pos = pos + 1

    print("This should be unreachable.")
    exit(1)


def lex_word(source: str, start: int) -> tuple[int, str]:
    pos = start
    while pos < len(source) and source[pos] not in SPACE_CHARACTERS | PUNCTUATION_CHARACTERS:
        pos = pos + 1
    return pos, source[start:pos]


def handle_preprocessor_directive(source: str, start: int, directive: str, loc: Loc) -> tuple[int, int, int, list[Token]]:
    pos = start

    if directive == "include":
        pos, seen, new_start = lex_spaces(source, pos)
        pos, path = lex_string(source, pos, loc.shift(pos - start))

        tokens = []

        # Prevent double includes
        if path not in FILES_INCLUDED:
            with open(path, "r") as f:
                include_src = f.read()
            tokens = lex(include_src, path)

            # Get rid of the extra EoF
            _ = tokens.pop()

        return pos, seen, new_start, tokens

    else:
        print(f"{loc} Error: Unrecognised directive '{directive}'.")
        exit(1)


# ---------------------------------------------------------------------------------------------------------------------
# Parser Implementation
#

class InstrType(IntEnum):
    PUSH_DAT = auto()
    PUSH_STR = auto()
    SYSCALL = auto()
    RESERVE = auto()
    GET_NTH = auto()
    RELEASE = auto()
    SDUMP = auto()
    WRITE = auto()
    READ = auto()
    WCH = auto()
    RCH = auto()
    PUSH = auto()
    SWAP = auto()
    DROP = auto()
    DUP = auto()
    SND = auto()
    TRD = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    EQ = auto()
    LT = auto()
    GT = auto()
    BASEP = auto()
    PRINT = auto()
    LABEL = auto()
    JMPF = auto()
    JMP = auto()
    CALL = auto()
    RET = auto()
    END = auto()


@dataclass
class Instr:
    opcode: InstrType
    operand: str | int | None = None


BUILTINS = {
    "<stack-dump>": Instr(InstrType.SDUMP),
    "syscall0": Instr(InstrType.SYSCALL, 0),
    "syscall1": Instr(InstrType.SYSCALL, 1),
    "syscall2": Instr(InstrType.SYSCALL, 2),
    "syscall3": Instr(InstrType.SYSCALL, 3),
    "syscall4": Instr(InstrType.SYSCALL, 4),
    "syscall5": Instr(InstrType.SYSCALL, 5),
    "syscall6": Instr(InstrType.SYSCALL, 6),
    "write-ch": Instr(InstrType.WCH),
    "read-ch": Instr(InstrType.RCH),
    "write": Instr(InstrType.WRITE),
    "read": Instr(InstrType.READ),
    "swap": Instr(InstrType.SWAP),
    "drop": Instr(InstrType.DROP),
    "dup": Instr(InstrType.DUP),
    "2nd": Instr(InstrType.SND),
    "3rd": Instr(InstrType.TRD),
    "not": Instr(InstrType.NOT),
    "and": Instr(InstrType.AND),
    "or": Instr(InstrType.OR),
    "+": Instr(InstrType.ADD),
    "-": Instr(InstrType.SUB),
    "*": Instr(InstrType.MUL),
    "/": Instr(InstrType.DIV),
    "%": Instr(InstrType.MOD),
    "=": Instr(InstrType.EQ),
    "<": Instr(InstrType.LT),
    ">": Instr(InstrType.GT),
    ".": Instr(InstrType.PRINT),
    "#": Instr(InstrType.BASEP),
}


class Parsing(IntEnum):
    NULL = auto()
    WORD = auto()
    INLINE = auto()


CURRENTLY_PARSING = Parsing.NULL

STRINGS: list[str] = []

LOCALS: list[str] = []

DATA: dict[str, int] = {}
DATA_SIZE: int = 0

INLINES: dict[str, list[Instr]] = {}

# We use these to track functions, as well as manage any instances of
# use-before-declaration to ensure we don't call an undefined function
FUNCTION_MAP: dict[str, str] = {}
AWAITING_DEF: dict[str, list[Loc]] = {}


def get_function_label(name: str, loc: Loc) -> str:
    if name in FUNCTION_MAP:
        label = FUNCTION_MAP[name]
    else:
        label = make_label()
        FUNCTION_MAP[name] = label
        AWAITING_DEF[name] = []

    if name in AWAITING_DEF:
        AWAITING_DEF[name].append(loc)

    return label


def define_function(name: str, loc: Loc) -> str:
    if name in FUNCTION_MAP:
        # Is the name already defined?
        if name not in AWAITING_DEF:
            print(f"{loc} Error: WORD '{name}' has already been defined elsewhere.")
            exit(1)

        label = FUNCTION_MAP[name]
    elif name in INLINES or name in DATA:
        print(f"{loc} Error: WORD '{name}' has already been defined elsewhere.")
        exit(1)
    else:
        label = make_label()
        FUNCTION_MAP[name] = label

    if name in AWAITING_DEF:
        AWAITING_DEF.pop(name)

    return label


def parse(tokens: list[Token]) -> list[Instr]:
    pos, instrs = 0, []

    # Set up the main label
    main_label = get_function_label("main", tokens[-1].loc)

    # Use the main function as the entry point
    instrs.append(Instr(InstrType.CALL, main_label))
    instrs.append(Instr(InstrType.END))

    while pos < len(tokens) and tokens[pos].typ is not TokenType.EOF:
        token = tokens[pos]

        if token.lexeme == "def":
            pos, defn = parse_def(tokens, pos)
            instrs.extend(defn)

        elif token.lexeme == "data":
            pos = parse_data(tokens, pos)

        elif token.lexeme == "inline":
            pos = parse_inline_def(tokens, pos)

        else:
            print(f"{token.loc} Error: Expected KEYWORD 'def', 'data' or 'inline' but got {token.typ.name} '{token.lexeme}' instead.")
            exit(1)

    if "main" in AWAITING_DEF:
        print("Error: Program does not define an entry point. Please define the word 'main'.")

    if len(AWAITING_DEF) != 0:
        for name, locations in AWAITING_DEF.items():
            if name == "main": continue

            if len(locations) == 1:
                loc = locations[0]
                print(f"{loc} Error: The word '{name}' is never defined.")
            else:
                print(f"Error: The word '{name}' is never defined:")
                for loc in locations:
                    print(f"{loc} Note: '{name}' is used here.")

        exit(1)

    return instrs


def parse_def(tokens: list[Token], start: int) -> tuple[int, list[Instr]]:
    global CURRENTLY_PARSING
    CURRENTLY_PARSING = Parsing.WORD

    pos, instrs = start, []

    pos = expect_keyword("def", tokens, pos)

    pos, name = expect_word(tokens, pos)
    label = define_function(name, tokens[pos - 1].loc)
    instrs.append(Instr(InstrType.LABEL, label))

    pos = expect_keyword("is", tokens, pos)
    pos, body = parse_expression(tokens, pos)
    instrs.extend(body)

    pos = expect_keyword(";", tokens, pos)
    instrs.append(Instr(InstrType.RET))

    return pos, instrs


def parse_data(tokens: list[Token], start: int) -> int:
    pos = expect_keyword("data", tokens, start)

    name_index = pos
    pos, name = expect_word(tokens, pos)

    pos, size = expect_number(tokens, pos)
    pos = expect_keyword(";", tokens, pos)

    global AWAITING_DEF, DATA, DATA_SIZE, FUNCTION_MAP, INLINES

    if name in AWAITING_DEF:
        locations = AWAITING_DEF[name]
        if len(locations) == 1:
            print(f"{locations[0]} Error: Data '{name}' must be declared before use.")
            exit(1)

        print(f"Error: Data '{name}' must be declared before use.")
        for loc in locations:
            print(f"{loc} Note: '{name}' is used here.")
        exit(1)

    if name in DATA or name in FUNCTION_MAP or name in INLINES:
        print(f"{tokens[name_index].loc} Error: The name '{name}' is already defined elsewhere.")
        exit(1)

    DATA[name] = DATA_SIZE
    DATA_SIZE = DATA_SIZE + size

    return pos


def parse_inline_def(tokens: list[Token], start: int) -> int:
    global CURRENTLY_PARSING
    CURRENTLY_PARSING = Parsing.INLINE

    pos = expect_keyword("inline", tokens, start)

    name_index = pos
    pos, name = expect_word(tokens, pos)
    pos = expect_keyword("is", tokens, pos)

    pos, body = parse_expression(tokens, pos)
    pos = expect_keyword(";", tokens, pos)

    global AWAITING_DEF, DATA, FUNCTION_MAP, INLINES

    if name in AWAITING_DEF:
        locations = AWAITING_DEF[name]
        if len(locations) == 1:
            print(f"{locations[0]} Error: Inline definition '{name}' must be declared before use.")
            exit(1)

        print(f"Error: Inline definition '{name}' must be declared before use.")
        for loc in locations:
            print(f"{loc} Note: '{name}' is used here.")
        exit(1)

    if name in DATA or name in FUNCTION_MAP or name in INLINES:
        print(f"{tokens[name_index].loc} Error: The name '{name}' is already defined elsewhere.")
        exit(1)

    INLINES[name] = body
    return pos


def parse_expression(tokens: list[Token], start: int) -> tuple[int, list[Instr]]:
    pos, instrs = start, []

    global BUILTINS, DATA, LOCALS, INLINES, CURRENTLY_PARSING
    assert len(TokenType) == 6, "Make sure all token types are handled as necessary."

    while pos < len(tokens):
        token: Token = tokens[pos]

        # Builtin words can be handled by their respective instruction
        if token.typ is TokenType.WORD and token.lexeme in BUILTINS:
            instrs.append(BUILTINS[token.lexeme])
            pos = pos + 1

        # Words referring to a local push the index of the local onto the stack
        elif token.typ is TokenType.WORD and token.lexeme in LOCALS:
            instrs.append(Instr(InstrType.GET_NTH, LOCALS.index(token.lexeme) + 1))
            pos = pos + 1

        # Words referring to an inline definition have the body inlined
        elif token.typ is TokenType.WORD and token.lexeme in INLINES:
            instrs.extend(INLINES[token.lexeme])
            pos = pos + 1

        # Words referring to the data segment push a pointer to the start of their segment
        elif token.typ is TokenType.WORD and token.lexeme in DATA:
            instrs.append(Instr(InstrType.PUSH_DAT, DATA[token.lexeme]))
            pos = pos + 1

        # Non builtin words can be handled by a call instruction
        elif token.typ is TokenType.WORD:
            label = get_function_label(token.lexeme, token.loc)
            instrs.append(Instr(InstrType.CALL, label))
            pos = pos + 1

        elif token.typ is TokenType.KEYWORD and token.lexeme == "return":
            if CURRENTLY_PARSING != Parsing.WORD:
                print(f"{token.loc} Error: Cannot return from somewhere that is not a word.")
                exit(1)

            if len(LOCALS) > 0:
                instrs.append(Instr(InstrType.RELEASE, len(LOCALS)))
            instrs.append(Instr(InstrType.RET))
            pos = pos + 1

        # LOCAL ::= '->' <word> (',' <word>)* 'do' <exp> ';'
        elif token.typ is TokenType.KEYWORD and token.lexeme == "->":

            # Parse the one mandatory name
            pos = expect_keyword("->", tokens, pos)
            pos, name = expect_word(tokens, pos)

            # Parse the rest of the names listed
            names = [name]
            while pos < len(tokens) and tokens[pos].lexeme == ",":
                pos, name = expect_word(tokens, pos + 1)
                names.append(name)

            # Reserve one slot on the stack for each name
            new_locals_count = len(names)
            instrs.append(Instr(InstrType.RESERVE, new_locals_count))
            LOCALS = [*names, *LOCALS]

            # Parse the rest of the body and add in its instructions
            pos = expect_keyword("do", tokens, pos)
            pos, body = parse_expression(tokens, pos)
            instrs.extend(body)

            # Release the reserved stack space
            pos = expect_keyword(";", tokens, pos)
            instrs.append(Instr(InstrType.RELEASE, new_locals_count))
            LOCALS = LOCALS[new_locals_count:]

        # IF ::= 'if' <exp> 'then' <exp> ('else-if' <exp> 'then' <exp>)* ('else' <exp>)? ';'
        elif token.typ is TokenType.KEYWORD and token.lexeme == "if":

            # Parse out the condition and add its instructions
            pos = expect_keyword("if", tokens, pos)
            pos, cond = parse_expression(tokens, pos)
            instrs.extend(cond)

            # We need to be able to jump to the false case, so
            # create a label and hold onto it's text till we know
            # where the label needs to be placed
            false_branch = make_label()
            instrs.append(Instr(InstrType.JMPF, false_branch))

            # Parse out the body and add its instructions
            pos = expect_keyword("then", tokens, pos)
            pos, body = parse_expression(tokens, pos)
            instrs.extend(body)

            # Now that the body has ended, we need to add a jump
            # to wherever the if-statement ends, so make the label
            end_label = make_label()
            instrs.append(Instr(InstrType.JMP, end_label))

            # The false case starts here, so emit the label for it
            instrs.append(Instr(InstrType.LABEL, false_branch))

            while pos < len(tokens) and tokens[pos].lexeme == "else-if":

                # Parse the condition
                pos = expect_keyword("else-if", tokens, pos)
                pos, cond = parse_expression(tokens, pos)
                instrs.extend(cond)

                # Emit the conditional jump
                false_branch = make_label()
                instrs.append(Instr(InstrType.JMPF, false_branch))

                # Parse the body
                pos = expect_keyword("then", tokens, pos)
                pos, body = parse_expression(tokens, pos)
                instrs.extend(body)

                # Emit the jump to the end, reusing the label
                instrs.append(Instr(InstrType.JMP, end_label))

                # The false case starts here
                instrs.append(Instr(InstrType.LABEL, false_branch))

            if pos < len(tokens) and tokens[pos].lexeme == "else":
                # The final else has no condition so just parse body
                pos = expect_keyword("else", tokens, pos)
                pos, body = parse_expression(tokens, pos)
                instrs.extend(body)

            pos = expect_keyword(";", tokens, pos)
            instrs.append(Instr(InstrType.LABEL, end_label))

        # WHILE ::= 'while' <exp> 'do' <exp> ';'
        elif token.typ is TokenType.KEYWORD and token.lexeme == "while":

            # We need to be able to jump back to the condition, so make
            # a label and remember it, so we can jump to it later
            loop_start = make_label()
            instrs.append(Instr(InstrType.LABEL, loop_start))

            # Parse and emit the condition for the loop
            pos = expect_keyword("while", tokens, pos)
            pos, cond = parse_expression(tokens, pos)
            instrs.extend(cond)

            # Emit a jump to the end if the condition is false
            false_branch = make_label()
            instrs.append(Instr(InstrType.JMPF, false_branch))

            # Parse and emit the body of the loop
            pos = expect_keyword("do", tokens, pos)
            pos, body = parse_expression(tokens, pos)
            instrs.extend(body)

            # Emit a jump back to the start of the loop
            instrs.append(Instr(InstrType.JMP, loop_start))

            # Emit the label for the end of the loop
            pos = expect_keyword(";", tokens, pos)
            instrs.append(Instr(InstrType.LABEL, false_branch))

        # Numbers simply push their value onto the stack
        elif token.typ is TokenType.NUMBER:
            instrs.append(Instr(InstrType.PUSH, int(token.lexeme)))
            pos = pos + 1

        # String literals push a pointer to their base and their length onto the stack
        elif token.typ is TokenType.STRING:
            instrs.append(Instr(InstrType.PUSH, len(token.lexeme)))
            instrs.append(Instr(InstrType.PUSH_STR, sum(len(s) for s in STRINGS)))
            STRINGS.append(token.lexeme)
            pos = pos + 1

        # Anything that is none of the above is left to be handled
        # by the parent scope that called this function
        else:
            break

    return pos, instrs


def expect_keyword(lexeme: str, tokens: list[Token], index: int) -> int:
    if index >= len(tokens) - 1:
        print(f"{tokens[-1].loc} Error: Expected '{lexeme}' but got nothing instead.")
        exit(1)

    token = tokens[index]
    if token.lexeme != lexeme:
        print(f"{token.loc} Error: Expected '{lexeme}' but got {token.typ.name} '{token.lexeme}' instead.")
        exit(1)

    return index + 1


def expect_word(tokens: list[Token], index: int) -> tuple[int, str]:
    if not index < len(tokens):
        print(f"{tokens[-1].loc} Error: expected WORD but got nothing instead.")
        exit(1)

    token = tokens[index]
    if token.typ is not TokenType.WORD:
        print(f"{token.loc} Error: Expected WORD but got {token.typ.name} '{token.lexeme}' instead.")
        exit(1)

    return index + 1, token.lexeme


def expect_number(tokens: list[Token], index: int) -> tuple[int, int]:
    if not index < len(tokens):
        print(f"{tokens[-1].loc} Error: expected NUMBER but got nothing instead.")
        exit(1)

    token = tokens[index]
    if token.typ is not TokenType.NUMBER:
        print(f"{token.loc} Error: Expected NUMBER but got {token.typ.name} '{token.lexeme}' instead.")
        exit(1)

    return index + 1, int(token.lexeme)


LABEL = 0


def make_label() -> str:
    global LABEL

    cur = LABEL
    LABEL = LABEL + 1
    return f"L{cur}"


# ---------------------------------------------------------------------------------------------------------------------
# Interpreter Implementation
#

def interpret(instructions: list[Instr]) -> None:

    ip, heap = 0, bytearray(1024 * 1024)
    val_stack, ret_stack = [], []

    total_allocated = 0

    files: dict[int, int] = { 0: 0 , 1: 1, 2: 2 }
    next_file = 3

    # Add all the strings to the heap. Start at 1 because 0 should be Null
    strings_end = 1
    for string in STRINGS:
        heap[strings_end:strings_end+len(string)] = string.encode('utf-8')
        strings_end = strings_end + len(string)

    data_end = strings_end + DATA_SIZE

    # Find all the labels in the program and
    # record their index so that the interpreter
    # can jump to them via this lookup table
    jump_table = {
        instr.operand: ip + 1
        for ip, instr
        in enumerate(instructions)
        if instr.opcode is InstrType.LABEL
    }

    assert len(InstrType) == 36, "Make sure all instructions are handled as necessary."

    while ip < len(instructions):
        instr = instructions[ip]
        ip = ip + 1

        if instr.opcode is InstrType.PUSH:
            val_stack.append(instr.operand)

        elif instr.opcode is InstrType.PUSH_STR:
            val_stack.append(1 + instr.operand)

        elif instr.opcode is InstrType.PUSH_DAT:
            val_stack.append(strings_end + instr.operand)

        elif instr.opcode is InstrType.RESERVE:
            assert isinstance(instr.operand, int), "The Operand of Reserve must be an integer"
            # Push the top n items from the value stack to the return stack, in reverse order
            for _ in range(instr.operand):
                ret_stack.append(val_stack.pop())

        elif instr.opcode is InstrType.RELEASE:
            assert isinstance(instr.operand, int), "The Operand of Release must be an integer"
            # Discard the top n items of the return stack
            for _ in range(instr.operand):
                _ = ret_stack.pop()

        elif instr.opcode is InstrType.GET_NTH:
            assert isinstance(instr.operand, int), "The Operand of GetNth must be an integer"
            # Push the nth item from the top of the return stack to the value stack
            val_stack.append(ret_stack[-instr.operand])

        elif instr.opcode is InstrType.BASEP:
            val_stack.append(data_end)

        elif instr.opcode is InstrType.WRITE:
            address = val_stack.pop()
            value: int = val_stack.pop()
            heap[address:address+8] = value.to_bytes(8, "little")

        elif instr.opcode is InstrType.READ:
            address = val_stack.pop()
            value: int = int.from_bytes(heap[address:address+8], "little")
            val_stack.append(value)

        elif instr.opcode is InstrType.WCH:
            address = val_stack.pop()
            value: int = val_stack.pop()
            heap[address:address+1] = value.to_bytes(1, "little")

        elif instr.opcode is InstrType.RCH:
            address = val_stack.pop()
            value: int = int.from_bytes(heap[address:address+1], "little")
            val_stack.append(value)

        elif instr.opcode is InstrType.ADD:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(left + right)

        elif instr.opcode is InstrType.SUB:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(left - right)

        elif instr.opcode is InstrType.MUL:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(left * right)

        elif instr.opcode is InstrType.DIV:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(left // right)

        elif instr.opcode is InstrType.MOD:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(left % right)

        elif instr.opcode is InstrType.NOT:
            top = val_stack.pop()
            val_stack.append(int(not top))

        elif instr.opcode is InstrType.AND:
            left = val_stack.pop()
            right = val_stack.pop()
            val_stack.append(int(left and right))

        elif instr.opcode is InstrType.OR:
            left = val_stack.pop()
            right = val_stack.pop()
            val_stack.append(int(left or right))

        elif instr.opcode is InstrType.EQ:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(int(left == right))

        elif instr.opcode is InstrType.LT:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(int(left < right))

        elif instr.opcode is InstrType.GT:
            right = val_stack.pop()
            left = val_stack.pop()
            val_stack.append(int(left > right))

        elif instr.opcode is InstrType.SWAP:
            top = val_stack.pop()
            snd = val_stack.pop()
            val_stack.append(top)
            val_stack.append(snd)

        elif instr.opcode is InstrType.DROP:
            _ = val_stack.pop()

        elif instr.opcode is InstrType.DUP:
            top = val_stack[-1]
            val_stack.append(top)

        elif instr.opcode is InstrType.SND:
            val_stack.append(val_stack[-2])

        elif instr.opcode is InstrType.TRD:
            val_stack.append(val_stack[-3])

        elif instr.opcode is InstrType.LABEL:
            pass

        elif instr.opcode is InstrType.JMPF:
            top = val_stack.pop()
            if top == 0:
                ip = jump_table[instr.operand]

        elif instr.opcode is InstrType.JMP:
            ip = jump_table[instr.operand]

        elif instr.opcode is InstrType.CALL:
            ret_stack.append(ip)
            ip = jump_table[instr.operand]

        elif instr.opcode is InstrType.RET:
            ip = ret_stack.pop()

        elif instr.opcode is InstrType.PRINT:
            top = val_stack.pop()
            print(top)

        elif instr.opcode is InstrType.SYSCALL:
            assert isinstance(instr.operand, int), "Syscall operand must be an int."

            args = []
            for _ in range(1 + instr.operand):
                args.append(val_stack.pop())

            call, *args = args

            # Read
            if call == 0:
                fd, buf, l = args
                data = os.read(files[fd], l)
                heap[buf:buf+l] = data
                val_stack.append(len(data))

            # Write
            elif call == 1:
                fd, buf, l = args
                data = heap[buf:buf+l]
                wrote = os.write(files[fd], data)
                val_stack.append(wrote)

            # Open
            elif call == 2:
                # TODO: Actually make flags and mode do something
                ptr, flags, mode = args
                name = heap[ptr:heap.find(0, ptr)]
                fd = os.open(name, os.O_RDWR)
                files[next_file] = fd
                val_stack.append(next_file)
                next_file = next_file + 1

            # Close
            elif call == 3:
                fd, = args
                os.close(files[fd])
                # TODO: Don't just fake successful close
                val_stack.append(0)

            # MMap
            elif call == 9:
                addr, l, prot, flags, fd, offset = args

                if fd != -1:
                    print("Error: Currently cannot mmap actual files.")
                    exit(1)

                val_stack.append(data_end + total_allocated)
                total_allocated = total_allocated + l
                # print(total_allocated)

            # Exit
            elif call == 60:
                return

            else:
                print(f"Error: Unimplemented Syscall '{call}'.")
                exit(1)

        elif instr.opcode is InstrType.SDUMP:
            print(val_stack)

        elif instr.opcode is InstrType.END:
            return

        else:
            print(f"Error: unhandled opcode {instr.opcode.name}")
            exit(1)

        # print(instr.opcode.name, ret_stack, val_stack, heap[0:40], end="")
        # input()

    print(f"Error: The instructions provided never called END. This should not happen.")
    exit(1)


# ---------------------------------------------------------------------------------------------------------------------
# Compiler Implementation
#

class Backend:
    @staticmethod
    def _emit_all(file: IO, strings: list[str]) -> None:
        file.write("".join(line + "\n" for line in strings))

    @staticmethod
    def begin(file: IO) -> None:
        """Emits assembly prelude"""
        _ = file
        raise NotImplementedError("Backend must be a derived class representing a target")

    @staticmethod
    def end(file: IO) -> None:
        """Emits assembly postlude"""
        _ = file
        raise NotImplementedError("Backend must be a derived class representing a target")

    @staticmethod
    def emit_instruction(file: IO, instruction: Instr) -> None:
        """Emits a single instruction"""
        _, _ = file, instruction
        raise NotImplementedError("Backend must be a derived class representing a target")


def gen_code(ir: list[Instr], backend: Type[Backend], file: IO) -> None:
    backend.begin(file)
    for instruction in ir:
        backend.emit_instruction(file, instruction)
    backend.end(file)


class Linux_x86_64(Backend):
    @staticmethod
    def begin(file: IO) -> None:
        Backend._emit_all(file, [
            f"section .bss",
            f"data_base:",
            f"    resb {DATA_SIZE}",
            f"stack_base:",
            f"    resb 1024",
            f"heap_base:",
            f"    resb 1024*1024",

            f"section .rodata",
            f"str_table:",
            *(f"    db {','.join(f'0x{b:02X}' for b in s.encode('utf-8'))}" for s in STRINGS),

            f"section .text",
            f"global _start",
            f"_start:",
            f"    mov r15, stack_base",
        ])

    @staticmethod
    def end(file: IO) -> None:
        Backend._emit_all(file, [
            "_dot_op:",
            "    sub     rsp, 32",
            "    mov     rsi, rsp",
            "    mov     byte [rsi + 31], 10",
            "    lea     r8, [rsi + 30]",
            "    mov     rcx, 1",
            "    mov     rbx, 10",
            ".convert_loop:",
            "    xor     rdx, rdx",
            "    mov     rax, rdi",
            "    div     rbx",
            "    add     dl, '0'",
            "    mov     byte [r8], dl",
            "    dec     r8",
            "    inc     rcx",
            "    mov     rdi, rax",
            "    test    rax, rax",
            "    jne     .convert_loop",
            "    inc     r8",
            "    mov     rax, 1",
            "    mov     rdi, 1",
            "    mov     rsi, r8",
            "    mov     rdx, rcx",
            "    syscall",
            "    add     rsp, 32",
            "    ret",
        ])

    @staticmethod
    def emit_instruction(file: IO, instruction: Instr) -> None:
        opcode, operand = instruction.opcode, instruction.operand

        assert len(InstrType) == 36, "make sure to account for all instruction types"

        if opcode is InstrType.PUSH:
            Backend._emit_all(file, [
                f"; {operand}",
                f"    push    {operand}"
            ])

        elif opcode is InstrType.PUSH_STR:
            Backend._emit_all(file, [
                f"; str {operand}",
                f"    lea     rax, [rel str_table+{operand}]",
                f"    push    rax",
            ])

        elif opcode is InstrType.PUSH_DAT:
            Backend._emit_all(file, [
                f"; data {operand}",
                f"    lea     rax, [rel data_base+{operand}]",
                f"    push    rax",
            ])

        elif opcode is InstrType.RESERVE:
            assert isinstance(operand, int), "The Operand of Reserve must be an integer"
            Backend._emit_all(file, [
                f"; reserve {operand}",
            ])

            for _ in range(operand):
                Backend._emit_all(file, [
                    f"    pop     rax",
                    f"    mov     qword [r15], rax",
                    f"    add     r15, 8",
                ])

        elif opcode is InstrType.RELEASE:
            assert isinstance(operand, int), "The Operand of Release must be an integer"
            Backend._emit_all(file, [
                f"; release {operand}",
                f"    sub     r15, {8 * operand}",
            ])

        elif opcode is InstrType.GET_NTH:
            assert isinstance(operand, int), "The Operand of GetNth must be an integer"
            Backend._emit_all(file, [
                f"; get local {operand}",
                f"    mov     rax, [r15-{8 * operand}]",
                f"    push    rax",
            ])

        elif opcode is InstrType.BASEP:
            Backend._emit_all(file, [
                f"; base-pointer",
                f"    lea     rax, [rel heap_base]",
                f"    push    rax",
            ])

        elif opcode is InstrType.WRITE:
            Backend._emit_all(file, [
                f"; write",
                f"    pop     rcx",
                f"    pop     rax",
                f"    mov     qword [rcx], rax",
            ])

        elif opcode is InstrType.WCH:
            Backend._emit_all(file, [
                f"; write-ch",
                f"    pop     rcx",
                f"    pop     rax",
                f"    mov     byte [rcx], al",
            ])

        elif opcode is InstrType.READ:
            Backend._emit_all(file, [
                f"; read",
                f"    pop     rcx",
                f"    mov     rax, [rcx]",
                f"    push    rax",
            ])

        elif opcode is InstrType.RCH:
            Backend._emit_all(file, [
                f"; read-ch",
                f"    pop     rcx",
                f"    movzx   rax, byte [rcx]",
                f"    push rax",
            ])

        elif opcode is InstrType.LABEL:
            Backend._emit_all(file, [
                f"{operand}:"
            ])

        elif opcode is InstrType.DUP:
            Backend._emit_all(file, [
                f"; dup",
                f"    mov     rdx, [rsp]",
                f"    push    rdx"
            ])

        elif opcode is InstrType.SND:
            Backend._emit_all(file, [
                f"; 2nd",
                f"    mov     rdx, [rsp+8]",
                f"    push    rdx",
            ])

        elif opcode is InstrType.TRD:
            Backend._emit_all(file, [
                f"; 3rd",
                f"    mov     rdx, [rsp+16]",
                f"    push    rdx",
            ])

        elif opcode is InstrType.SWAP:
            Backend._emit_all(file, [
                f"; swap",
                f"    pop     rax",
                f"    pop     rdx",
                f"    push    rax",
                f"    push    rdx"
            ])

        elif opcode is InstrType.DROP:
            Backend._emit_all(file, [
                f"; drop",
                f"    pop     rax",
            ])

        elif opcode is InstrType.EQ:
            Backend._emit_all(file, [
                f"; eq",
                f"    pop     rbx",
                f"    pop     rax",
                f"    cmp     rax, rbx",
                f"    sete    al",
                f"    movzx   rax, al",
                f"    push    rax"
            ])

        elif opcode is InstrType.GT:
            Backend._emit_all(file, [
                f"; gt",
                f"    pop     rbx",
                f"    pop     rax",
                f"    cmp     rax, rbx",
                f"    setg    al",
                f"    movzx   rax, al",
                f"    push    rax"
            ])

        elif opcode is InstrType.LT:
            Backend._emit_all(file, [
                f"; lt",
                f"    pop     rbx",
                f"    pop     rax",
                f"    cmp     rax, rbx",
                f"    setl    al",
                f"    movzx   rax, al",
                f"    push    rax"
            ])

        elif opcode is InstrType.OR:
            Backend._emit_all(file, [
                f"; or",
                f"    pop     rbx",
                f"    pop     rax",
                f"    or      rax, rbx",
                f"    push    rax",
            ])

        elif opcode is InstrType.AND:
            Backend._emit_all(file, [
                f"; and",
                f"    pop     rbx",
                f"    pop     rax",
                f"    and     rax, rbx",
                f"    push    rax",
            ])

        elif opcode is InstrType.NOT:
            Backend._emit_all(file, [
                f"; not",
                f"    pop     rax",
                f"    xor     rax, 1",
                f"    push    rax"
            ])

        elif opcode is InstrType.JMPF:
            Backend._emit_all(file, [
                f"; jmpf",
                f"    pop     rax",
                f"    cmp     rax, 0",
                f"    je      {operand}"
            ])

        elif opcode is InstrType.CALL:
            l = make_label()
            Backend._emit_all(file, [
                f"; call",
                f"    lea     rax, [rel {l}]",
                f"    mov     [r15], rax",
                f"    add     r15, 8",
                f"    jmp     {operand}",
                f"{l}:"
            ])

        elif opcode is InstrType.RET:
            Backend._emit_all(file, [
                f"; ret",
                f"    sub r15, 8",
                f"    mov rax, [r15]",
                f"    jmp rax",
            ])

        elif opcode is InstrType.JMP:
            Backend._emit_all(file, [
                f"; jmp",
                f"    jmp     {operand}"
            ])

        elif opcode is InstrType.PRINT:
            Backend._emit_all(file, [
                f"; print",
                f"    pop     rdi",
                f"    call    _dot_op"
            ])

        elif opcode is InstrType.ADD:
            Backend._emit_all(file, [
                f"; add",
                f"    pop     rbx",
                f"    pop     rax",
                f"    add     rax, rbx",
                f"    push    rax"
            ])

        elif opcode is InstrType.SUB:
            Backend._emit_all(file, [
                f"; sub",
                f"    pop     rbx",
                f"    pop     rax",
                f"    sub     rax, rbx",
                f"    push    rax",
            ])

        elif opcode is InstrType.MUL:
            Backend._emit_all(file, [
                f"; mul",
                f"    pop     rbx",
                f"    pop     rax",
                f"    imul    rax, rbx",
                f"    push    rax",
            ])

        elif opcode is InstrType.DIV:
            Backend._emit_all(file, [
                f"; div",
                f"    pop     rbx",
                f"    pop     rax",
                f"    xor     rdx, rdx",
                f"    idiv    rbx",
                f"    push    rax",
            ])

        elif opcode is InstrType.MOD:
            Backend._emit_all(file, [
                f"; mod",
                f"    pop     rbx",
                f"    pop     rax",
                f"    xor     rdx, rdx",
                f"    idiv    rbx",
                f"    push    rdx",
            ])

        elif opcode is InstrType.SYSCALL:
            if not isinstance(operand, int):
                print("Compiler has failed, syscall arg should always be an int.")
                exit(1)

            Backend._emit_all(file, [
                f"; syscall {operand}",
                f"    pop     rax",
            ])

            if operand > 0: Backend._emit_all(file, [f"    pop     rdi"])
            if operand > 1: Backend._emit_all(file, [f"    pop     rsi"])
            if operand > 2: Backend._emit_all(file, [f"    pop     rdx"])
            if operand > 3: Backend._emit_all(file, [f"    pop     r10"])
            if operand > 4: Backend._emit_all(file, [f"    pop     r8"])
            if operand > 5: Backend._emit_all(file, [f"    pop     r9"])

            Backend._emit_all(file, [
                f"    syscall",
                f"    push    rax",
            ])

        elif opcode is InstrType.END:
            Backend._emit_all(file, [
                f"; end",
                f"    mov     rax, 60",
                f"    xor     rdi, rdi",
                f"    syscall",
            ])

        elif opcode is InstrType.SDUMP:
            print("Stack Dump is unimplemented for compiler right now.")
            exit(1)

        else:
            print(f"Unimplemented Instruction: {opcode.name}")
            exit(1)


# ---------------------------------------------------------------------------------------------------------------------
# Debug Helpers
#

def dump_ir(instructions: list[Instr]) -> None:
    print("start:")
    for i, instr in enumerate(instructions):
        if instr.opcode is InstrType.LABEL:
            # If we aren't in a block of labels, print a newline for separation
            if i != 0 and instructions[i - 1].opcode is not InstrType.LABEL:
                print()
            print(f"{instr.operand}:")
            continue

        print(f"  [{i:04}]    {instr.opcode.name:9}{instr.operand if instr.operand is not None else ''}")


# ---------------------------------------------------------------------------------------------------------------------
# Entry Point
#

if __name__ == "__main__":
    main()
