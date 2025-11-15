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
    WORD = auto()
    PUNC = auto()
    EOF = auto()


@dataclass
class Token:
    typ: TokenType
    lexeme: str
    loc: tuple[int, int]


KEY_WORDS = {"def", "is", "if", "then", "if-else", "else", "do", "while"}
SPACE_CHARACTERS = {" ", "\n", "\t"}
PUNCTUATION_CHARACTERS = {";", ",", "[", "]"}


def make_token(typ: TokenType, lexeme: str, line: int, line_start: int, end: int) -> Token:
    column = end - line_start - len(lexeme)
    return Token(typ, lexeme, (line, column))


def lex(source: str) -> list[Token]:
    pos = 0
    line_count, line_start = 0, 0

    tokens = []
    while pos < len(source):
        char = source[pos]

        if char in SPACE_CHARACTERS:
            pos, lines_passed, new_start = lex_spaces(source, pos)
            if lines_passed > 0:
                line_count = line_count + lines_passed
                line_start = new_start

        elif char in PUNCTUATION_CHARACTERS:
            pos = pos + 1
            tokens.append(make_token(TokenType.PUNC, char, line_count, line_start, pos))

        else:
            pos, lexeme = lex_word(source, pos)
            typ = TokenType.KEY_WORD if lexeme in KEY_WORDS else TokenType.WORD
            tokens.append(make_token(typ, lexeme, line_count, line_start, pos))

    tokens.append(make_token(TokenType.EOF, "EoF", line_count, line_start, pos))
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


def lex_word(source: str, start: int) -> tuple[int, str]:
    pos = start
    while pos < len(source) and source[pos] not in SPACE_CHARACTERS | PUNCTUATION_CHARACTERS:
        pos = pos + 1
    return pos, source[start:pos]


if __name__ == "__main__":
    main(argv)
