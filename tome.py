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

    tokens = lex(source)
    for tok in tokens:
        print(tok)


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
    row: int
    col: int

    def shift(self, d: int) -> 'Loc':
        return Loc(self.row, self.col + d)

    def __repr__(self) -> str:
        return f"[{self.row}:{self.col}]"


@dataclass
class Token:
    typ: TokenType
    lexeme: str
    loc: Loc


KEY_WORDS = {"def", "is", "if", "then", "else-if", "else", "do", "while"}
SPACE_CHARACTERS = {" ", "\n", "\t"}
PUNCTUATION_CHARACTERS = {";", ",", "[", "]"}


def lex(source: str) -> list[Token]:
    pos = 0
    line_count, line_start = 0, 0

    tokens = []
    while pos < len(source):
        char = source[pos]

        # The location is the row (aka line) and the
        # column (how far we are past the start of the line)
        loc = Loc(line_count, pos - line_start)

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

    tokens.append(Token(TokenType.EOF, "EoF", Loc(line_count, pos)))
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

            if next_ == "\\" or next_ == "\"":
                string_content += next_

            elif next_ == "n":
                string_content += "\n"

            elif next_ == "\t":
                string_content += "\t"

            else:
                print(f"{loc.shift(pos - start + 1)} Error: Unrecognised escape sequence.")
                exit(1)

            pos = pos + 2

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


if __name__ == "__main__":
    main(argv)
