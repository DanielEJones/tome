from unittest import TestCase, main

from src.lexer import Lexer, TokenType


class TestLexingWord(TestCase):
    def test_can_lex_words(self) -> None:
        lexemes = ["normal", "<and>", "also-some", ":special", "0nes", "(here)"]
        tks = Lexer("test", " ".join(lexemes)).lex()[:-1]  # strip off the EoF
        self.assertEqual(lexemes, [tk.lexeme for tk in tks], "Should be able to split by whitespace")
        self.assertTrue(all(tk.typ is TokenType.WORD for tk in tks), "All of these tokens should be words")

    def test_identifies_keywords(self) -> None:
        key_words = ["def", "is", "if", "then", "else-if", "else", "while", "do", "->"]
        tks = Lexer("test", " ".join(key_words)).lex()[:-1]
        self.assertTrue(all(tk.typ is TokenType.KEYWORD for tk in tks), "These should be keywords")


class TestLexingNumber(TestCase):
    def test_can_lex_number(self) -> None:
        nums = ["0", "1", "01", "123454567456789"]
        tks = Lexer("test", " ".join(nums)).lex()[:-1]
        self.assertTrue(all(tk.typ is TokenType.NUMBER for tk in tks))


class TestLexingPunctuation(TestCase):
    def test_can_lex_punctuation_with_space(self) -> None:
        tks = Lexer("test", "def add is + ;").lex()
        self.assertEqual(TokenType.PUNCTUATION, tks[4].typ, "Semicolon should be punctuation")

    def test_can_lex_punctuation_without_space(self) -> None:
        tks = Lexer("test", "def add is +;").lex()
        self.assertEqual(TokenType.PUNCTUATION, tks[4].typ, "Semicolon without leading space should be punctuation")


class TestLexingStrings(TestCase):
    def test_can_lex_simple_strings(self) -> None:
        tks = Lexer("test", '"hello, world!"').lex()
        self.assertEqual("hello, world!", tks[0].lexeme, "Should be able to lex a string")

    def test_can_lex_escape_sequence(self) -> None:
        tks = Lexer("test", """ "hello, \\"world\\"!" """).lex()
        self.assertEqual('hello, \\"world\\"!', tks[0].lexeme, "Should be able to lex an escape sequence")

    def test_errors_on_bad_escapes(self) -> None:
        lexer = Lexer("test", """ "\\f " """); lexer.lex()
        self.assertEqual(1, len(lexer.errors), "An unknown escape sequence should produce an error")
        self.assertEqual(3, lexer.errors[0][0].col, "The error should point to the escaped character")

    def test_errors_on_non_terminating(self) -> None:
        lexer = Lexer("test", """ "hello, world! """); lexer.lex()
        self.assertEqual(1, len(lexer.errors), "An unterminated string should produce an error")
        self.assertEqual(1, lexer.errors[0][0].col, "The error should point to the start of the string")


class TestLexingComments(TestCase):
    def test_comments_are_ignored(self) -> None:
        tks = Lexer("test", "-- this is a comment").lex()
        self.assertEqual(1, len(tks), "Comments should not produce tokens")

    def test_inline_comments_are_ignored(self) -> None:
        tks = Lexer("test", "def add is + ; -- this is a comment").lex()
        self.assertEqual(6, len(tks), "Comments should not produce tokens")


class TestLexingWhitespace(TestCase):
    def test_newlines_are_counted(self) -> None:
        tks = Lexer("test", "0\n1\n2\n3").lex()[:-1]
        self.assertTrue(all(tk.loc.row == i for i, tk in enumerate(tks)), "Each newline should increment row")

    def test_col_is_correctly_calculated(self) -> None:
        tks = Lexer("test", "0\n 1\n  2\n   3").lex()[:-1]
        self.assertTrue(all(tk.loc.col == i for i, tk in enumerate(tks)), "Each newline should reset col")


if __name__ == '__main__':
    main()
