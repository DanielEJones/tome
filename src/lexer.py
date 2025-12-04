from __future__ import annotations

from enum import IntEnum, auto
from dataclasses import dataclass


@dataclass
class Token:
    typ: TokenType
    lexeme: str
    loc: Loc

    def __repr__(self) -> str:
        lex = "" if not self.lexeme else f" '{self.lexeme}'"
        return f"<{self.loc} {self.typ.name}{lex}>"


class TokenType(IntEnum):
    WORD = auto()
    KEYWORD = auto()
    PUNCTUATION = auto()
    STRING = auto()
    NUMBER = auto()
    EOF = auto()


@dataclass
class Loc:
    file: str
    row: int
    col: int

    def __repr__(self) -> str:
        return f"[{self.file} @ {self.row}:{self.col}]"


class Lexer:
    def __init__(self, file_name: str, source: str) -> None:
        self._file_name, self._source = file_name, source
        self._pos, self._line, self._line_start = 0, 0, 0
        self.errors: list[tuple[Loc, str]] = []

    def lex(self) -> list[Token]:
        tokens = []
        while token := self._lex_one():
            tokens.append(token)
        tokens.append(Token(TokenType.EOF, "", self._loc()))
        return tokens

    def _lex_one(self) -> Token | None:
        self._skip_spaces()
        loc = self._loc()

        if not self._has_more():
            return None

        if punc := self._lex_punc():
            return Token(TokenType.PUNCTUATION, punc, loc)

        if string := self._lex_string():
            return Token(TokenType.STRING, string, loc)

        word = self._lex_word()
        if self.is_keyword(word):
            typ = TokenType.KEYWORD

        elif self.is_number(word):
            typ = TokenType.NUMBER

        else:
            typ = TokenType.WORD

        return Token(typ, word, loc)

    def _lex_punc(self) -> str | None:
        if self._has_more() and self._current() in {",", ";"}:
            lexeme = self._current()
            self._pos += 1
            return lexeme
        return None

    def _lex_string(self) -> str | None:
        loc = self._loc()
        if not (self._has_more() and self._current() == "\""):
            return None

        self._pos += 1
        start = self._pos

        while self._has_more():
            current = self._current()
            self._pos += 1

            if current == "\\":
                self._handle_escape()

            elif current == "\"":
                return self._source[start:self._pos-1]

        self._error_at(loc, "Unterminated string literal.")
        return None

    def _lex_word(self) -> str:
        start = self._pos

        while self._has_more() and not (self._is_space() or self._is_punctuation()):
            self._pos += 1

        return self._source[start:self._pos]

    def _skip_spaces(self) -> None:
        while True:
            while self._has_more() and self._is_space():
                if self._current() == "\n":
                    self._line += 1
                    self._line_start = self._pos + 1
                self._pos += 1

            if self._is_comment():
                while self._has_more() and not self._current() == "\n":
                    self._pos += 1
                continue

            return

    def _handle_escape(self) -> None:
        if self._has_more() and self._current() in {"t", "n", "0", "\"", "\'", "\\"}:
            self._pos += 1
            return

        if not self._has_more():
            self._error("Unterminated escape sequence.")
        else:
            self._error("Unrecognised escape sequence.")

        self._pos += 1

    def _current(self) -> str:
        return self._source[self._pos]

    def _has_more(self) -> bool:
        return self._pos < len(self._source)

    def _is_space(self) -> bool:
        return self._current() in {" ", "\t", "\n"}

    def _is_punctuation(self) -> bool:
        return self._current() in {",", ";"}

    def _is_comment(self) -> bool:
        return self._pos + 1 < len(self._source) and self._source[self._pos:self._pos+2] == "--"

    @staticmethod
    def is_keyword(word: str) -> bool:
        return word in {"def", "is", "if", "then", "else-if", "else", "while", "do", "->"}

    @staticmethod
    def is_number(word: str) -> bool:
        return all(ch.isdigit() for ch in word)

    def _loc(self) -> Loc:
        return Loc(self._file_name, self._line, self._pos - self._line_start)

    def _error_at(self, loc: Loc, message: str) -> None:
        self.errors.append((loc, message))

    def _error(self, message: str) -> None:
        self.errors.append((self._loc(), message))
