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

    print("_[ IP ]____MNEM____OP_")
    for i, instr in enumerate(instrs):
        print(f" [{i:04}]    {instr.opcode.name:8}{instr.operand if instr.operand is not None else ''}")


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

        # TODO: Maybe one day we will need punctuation >1 chars long
        elif char in PUNCTUATION_CHARACTERS:
            pos = pos + 1
            tokens.append(Token(TokenType.PUNC, char, loc))

        elif char.isdigit():
            pos, lexeme = lex_number(source, pos)
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
    PUSH = auto()
    DUP = auto()
    ADD = auto()
    SUB = auto()
    EQ = auto()
    PRINT = auto()
    JMPF = auto()
    JMP = auto()
    END = auto()


@dataclass
class Instr:
    opcode: InstrType
    operand: str | int | None = None


BUILTINS = {
    "dup": Instr(InstrType.DUP),
    "+": Instr(InstrType.ADD),
    "-": Instr(InstrType.SUB),
    "=": Instr(InstrType.EQ),
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


def parse_expression(tokens: list[Token], start: int, len_so_far: int = 0) -> tuple[int, list[Instr]]:
    pos, instrs = start, []

    while pos < len(tokens):
        token = tokens[pos]

        # Builtin words can be handled by their respective instruction
        if token.typ is TokenType.WORD and token.lexeme in BUILTINS:
            instrs.append(BUILTINS[token.lexeme])
            pos = pos + 1

        # Non builtin words can be handled by a call instruction
        elif token.typ is TokenType.WORD:
            print(f"{token.loc} Error: User-defined words are currently unsupported.")
            exit(1)

        # IF ::= 'if' <exp> 'then' <exp> ('else-if' <exp> 'then' <exp>)* ('else' <exp>)? ';'
        elif token.typ is TokenType.KEY_WORD and token.lexeme == "if":
            pos = expect_keyword("if", tokens, pos)
            pos, cond = parse_expression(tokens, pos, len_so_far + len(instrs))

            # We can emit the condition now, but we have to wait to emit
            # any further instructions until we know where to jump to
            instrs.extend(cond)

            # Instead we can emit 'holes' to patch and record the index
            false_branch = len(instrs)
            instrs.append(Instr(InstrType.JMPF, None))

            pos = expect_keyword("then", tokens, pos)
            pos, body = parse_expression(tokens, pos, len_so_far + len(instrs))

            instrs.extend(body)

            # We also need to maintain a list of holes to be filled once
            # we reach an else branch or fall off the end
            end_jmp = len(instrs)
            instrs.append(Instr(InstrType.JMP, None))
            end_patch_list = [end_jmp]

            instrs[false_branch].operand = len_so_far + len(instrs)

            # else-if blocks are largely the same as the above
            while pos < len(tokens) and tokens[pos].lexeme == "else-if":
                pos = expect_keyword("else-if", tokens, pos)
                pos, cond = parse_expression(tokens, pos, len_so_far + len(instrs))

                instrs.extend(cond)

                false_branch = len(instrs)
                instrs.append(Instr(InstrType.JMPF, None))

                pos = expect_keyword("then", tokens, pos)
                pos, body = parse_expression(tokens, pos, len_so_far + len(instrs))

                instrs.extend(body)

                end_jmp = len(instrs)
                instrs.append(Instr(InstrType.JMP, None))
                end_patch_list.append(end_jmp)

                instrs[false_branch].operand = len_so_far + len(instrs)

            if pos < len(tokens) and tokens[pos].lexeme == "else":
                pos = expect_keyword("else", tokens, pos)
                pos, body = parse_expression(tokens, pos, len_so_far + len(instrs))

                instrs.extend(body)

            # Go through and fill all the holes we left
            # TODO: If we only have an if and no else, we can remove the hole instead
            for patch_index in end_patch_list:
                instrs[patch_index].operand = len_so_far + len(instrs)

            pos = expect_keyword(";", tokens, pos)

        # WHILE ::= 'while' <exp> 'do' <exp> ';'
        elif token.typ is TokenType.KEY_WORD and token.lexeme == "while":
            print(f"{token.loc} Error: 'while' statements are currently unsupported.")
            exit(1)

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


if __name__ == "__main__":
    main(["tome", "./examples/if_else.tome"])
