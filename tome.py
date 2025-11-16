from sys import argv, exit
from enum import IntEnum, auto
from dataclasses import dataclass


def main(args: list[str]) -> None:
    name, *rest = args

    if len(rest) < 1:
        print(f"Error: Invalid command.")
        print(f"Usage: {name} <filepath>")
        exit(1)

    file_name, *_ = rest
    do_compile(file_name)


def do_compile(path: str) -> None:
    with open(path, "r") as source_file:
        source = source_file.read()

    tokens = lex(source, path)
    instrs = parse(tokens)

    print("--- BYTECODE ---")
    dump_ir(instrs)

    print("\n--- RESULT ---")
    interpret(instrs)


# ---------------------------------------------------------------------------------------------------------------------
# Lexer Implementation
#

class TokenType(IntEnum):
    KEY_WORD = auto()
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
        return f"[{self.file} @ {self.row}:{self.col}]"


@dataclass
class Token:
    typ: TokenType
    lexeme: str
    loc: Loc


KEY_WORDS = {"def", "is", "if", "then", "else-if", "else", "do", "while"}
SPACE_CHARACTERS = {" ", "\n", "\t"}
PUNCTUATION_CHARACTERS = {";", ",", "[", "]"}


def lex(source: str, file_name: str) -> list[Token]:
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

        elif char == "\"":
            pos, lexeme = lex_string(source, pos, loc)
            tokens.append(Token(TokenType.STRING, lexeme, loc))

        # Anything that isn't specifically special should be treated as a plain
        # word, unless it matches a keyword from the list
        else:
            pos, lexeme = lex_word(source, pos)
            typ = TokenType.KEY_WORD if lexeme in KEY_WORDS else TokenType.WORD
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

            elif next_ == "\t":
                string_content += "\t"

            else:
                print(f"{loc.shift(pos - start + 1)} Error: Unrecognised escape sequence.")
                exit(1)

        elif char == "\n":
            print(f"{loc} Error: Unterminated string.")
            exit(1)

        else:
            string_content += char
            pos = pos + 1


def lex_word(source: str, start: int) -> tuple[int, str]:
    pos = start
    while pos < len(source) and source[pos] not in SPACE_CHARACTERS | PUNCTUATION_CHARACTERS:
        pos = pos + 1
    return pos, source[start:pos]


# ---------------------------------------------------------------------------------------------------------------------
# Parser Implementation
#

class InstrType(IntEnum):
    SDUMP = auto()
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
    PRINT = auto()
    LABEL = auto()
    JMPF = auto()
    JMP = auto()
    END = auto()


@dataclass
class Instr:
    opcode: InstrType
    operand: str | int | None = None


BUILTINS = {
    "<stack-dump>": Instr(InstrType.SDUMP),
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
}


def parse(tokens: list[Token]) -> list[Instr]:
    pos, instrs = parse_expression(tokens, 0)

    last = tokens[pos]
    if last.typ is not TokenType.EOF:
        print(f"{last.loc} Error: Unexpected {last.typ.name} '{last.lexeme}'.")
        exit(1)

    instrs.append(Instr(InstrType.END))
    return instrs


def parse_expression(tokens: list[Token], start: int) -> tuple[int, list[Instr]]:
    pos, instrs = start, []

    assert len(TokenType) == 6, "Make sure all token types are handled as necessary."

    while pos < len(tokens):
        token = tokens[pos]

        # Builtin words can be handled by their respective instruction
        if token.typ is TokenType.WORD and token.lexeme in BUILTINS:
            instrs.append(BUILTINS[token.lexeme])
            pos = pos + 1

        # Non builtin words can be handled by a call instruction
        elif token.typ is TokenType.WORD:
            print(f"{token.loc} Error: User-defined words are currently unsupported: Unknown word '{token.lexeme}'.")
            exit(1)

        # IF ::= 'if' <exp> 'then' <exp> ('else-if' <exp> 'then' <exp>)* ('else' <exp>)? ';'
        elif token.typ is TokenType.KEY_WORD and token.lexeme == "if":

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
        elif token.typ is TokenType.KEY_WORD and token.lexeme == "while":

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

        # Anything that is none of the above is left to be handled
        # by the parent scope that called this function
        else:
            break

    return pos, instrs


def expect_keyword(lexeme: str, tokens: list[Token], index: int) -> int:
    if index >= len(tokens) - 1:
        print(f"{tokens[-1].loc} Error: Expected '{lexeme}' but got nothing.")
        exit(1)

    token = tokens[index]
    if token.lexeme != lexeme:
        print(f"{token.loc} Error: Expected '{lexeme}' but got {token.typ.name} '{token.lexeme}' instead.")
        exit(1)

    return index + 1


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
    ip, stack = 0, []

    # Find all the labels in the program and
    # record their index so that the interpreter
    # can jump to them via this lookup table
    jump_table: dict[str, int] = {
        instr.operand: ip + 1
        for ip, instr
        in enumerate(instructions)
        if instr.opcode is InstrType.LABEL
    }

    assert len(InstrType) == 23, "Make sure all instructions are handled as necessary."

    while ip < len(instructions):
        instr = instructions[ip]
        ip = ip + 1

        if instr.opcode is InstrType.PUSH:
            stack.append(instr.operand)

        elif instr.opcode is InstrType.ADD:
            right = stack.pop()
            left = stack.pop()
            stack.append(left + right)

        elif instr.opcode is InstrType.SUB:
            right = stack.pop()
            left = stack.pop()
            stack.append(left - right)

        elif instr.opcode is InstrType.MUL:
            right = stack.pop()
            left = stack.pop()
            stack.append(left * right)

        elif instr.opcode is InstrType.DIV:
            right = stack.pop()
            left = stack.pop()
            stack.append(left // right)

        elif instr.opcode is InstrType.MOD:
            right = stack.pop()
            left = stack.pop()
            stack.append(left % right)

        elif instr.opcode is InstrType.NOT:
            top = stack.pop()
            stack.append(int(not top))

        elif instr.opcode is InstrType.AND:
            left = stack.pop()
            right = stack.pop()
            stack.append(int(left and right))

        elif instr.opcode is InstrType.OR:
            left = stack.pop()
            right = stack.pop()
            stack.append(int(left or right))

        elif instr.opcode is InstrType.EQ:
            right = stack.pop()
            left = stack.pop()
            stack.append(int(left == right))

        elif instr.opcode is InstrType.LT:
            right = stack.pop()
            left = stack.pop()
            stack.append(int(left < right))

        elif instr.opcode is InstrType.GT:
            right = stack.pop()
            left = stack.pop()
            stack.append(int(left > right))

        elif instr.opcode is InstrType.SWAP:
            top = stack.pop()
            snd = stack.pop()
            stack.append(top)
            stack.append(snd)

        elif instr.opcode is InstrType.DROP:
            _ = stack.pop()

        elif instr.opcode is InstrType.DUP:
            top = stack[-1]
            stack.append(top)

        elif instr.opcode is InstrType.SND:
            stack.append(stack[-2])

        elif instr.opcode is InstrType.TRD:
            stack.append(stack[-3])

        elif instr.opcode is InstrType.LABEL:
            pass

        elif instr.opcode is InstrType.JMPF:
            top = stack.pop()
            if top == 0:
                ip = jump_table[instr.operand]

        elif instr.opcode is InstrType.JMP:
            ip = jump_table[instr.operand]

        elif instr.opcode is InstrType.PRINT:
            top = stack.pop()
            print(top)

        elif instr.opcode is InstrType.SDUMP:
            print(stack)

        elif instr.opcode is InstrType.END:
            return

        else:
            print(f"Error: unhandled opcode {instr.opcode.name}")
            exit(1)

    print(f"Error: The instructions provided never called END. This should not happen.")
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

        print(f"  [{i:04}]    {instr.opcode.name:8}{instr.operand if instr.operand is not None else ''}")


# ---------------------------------------------------------------------------------------------------------------------
# Entry Point
#

if __name__ == "__main__":
    main(argv)
