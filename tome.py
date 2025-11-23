from enum import IntEnum, auto
from typing import Type, TextIO
from dataclasses import dataclass
from subprocess import run

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

        elif char == "\'":
            pos, lexeme = lex_char(source, pos, loc)
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


# ---------------------------------------------------------------------------------------------------------------------
# Parser Implementation
#

class InstrType(IntEnum):
    PUSH_STR = auto()
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
    END = auto()


@dataclass
class Instr:
    opcode: InstrType
    operand: str | int | None = None


BUILTINS = {
    "<stack-dump>": Instr(InstrType.SDUMP),
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


STRINGS: list[str] = []


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
        token: Token = tokens[pos]

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

        # String literals push a pointer to their base and their length onto the stack
        elif token.typ is TokenType.STRING:
            instrs.append(Instr(InstrType.PUSH_STR, sum(len(s) for s in STRINGS)))
            instrs.append(Instr(InstrType.PUSH, len(token.lexeme)))
            STRINGS.append(token.lexeme)
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
    ip, stack, heap = 0, [], bytearray(1024 * 1024)

    # Add all the strings to the heap
    strings_end = 0
    for string in STRINGS:
        heap[strings_end:strings_end+len(string)] = string.encode('utf-8')
        strings_end = strings_end + len(string)

    # Find all the labels in the program and
    # record their index so that the interpreter
    # can jump to them via this lookup table
    jump_table = {
        instr.operand: ip + 1
        for ip, instr
        in enumerate(instructions)
        if instr.opcode is InstrType.LABEL
    }

    assert len(InstrType) == 29, "Make sure all instructions are handled as necessary."

    while ip < len(instructions):
        instr = instructions[ip]
        ip = ip + 1

        if instr.opcode is InstrType.PUSH:
            stack.append(instr.operand)

        elif instr.opcode is InstrType.PUSH_STR:
            stack.append(instr.operand)

        elif instr.opcode is InstrType.BASEP:
            stack.append(strings_end)

        elif instr.opcode is InstrType.WRITE:
            address = stack.pop()
            value: int = stack.pop()
            heap[address:address+8] = value.to_bytes(8, "little")

        elif instr.opcode is InstrType.READ:
            address = stack.pop()
            value: int = int.from_bytes(heap[address:address+8], "little")
            stack.append(value)

        elif instr.opcode is InstrType.WCH:
            address = stack.pop()
            value: int = stack.pop()
            heap[address:address+1] = value.to_bytes(1, "little")

        elif instr.opcode is InstrType.RCH:
            address = stack.pop()
            value: int = int.from_bytes(heap[address:address+1], "little")
            stack.append(value)

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
# Compiler Implementation
#

class Backend:
    @staticmethod
    def _emit_all(file: TextIO, strings: list[str]) -> None:
        file.write("".join(line + "\n" for line in strings))

    @staticmethod
    def begin(file: TextIO) -> None:
        """Emits assembly prelude"""
        _ = file
        raise NotImplementedError("Backend must be a derived class representing a target")

    @staticmethod
    def end(file: TextIO) -> None:
        """Emits assembly postlude"""
        _ = file
        raise NotImplementedError("Backend must be a derived class representing a target")

    @staticmethod
    def emit_instruction(file: TextIO, instruction: Instr) -> None:
        """Emits a single instruction"""
        _, _ = file, instruction
        raise NotImplementedError("Backend must be a derived class representing a target")


def gen_code(ir: list[Instr], backend: Type[Backend], file: TextIO) -> None:
    backend.begin(file)
    for instruction in ir:
        backend.emit_instruction(file, instruction)
    backend.end(file)


class Linux_x86_64(Backend):
    @staticmethod
    def begin(file: TextIO) -> None:
        Backend._emit_all(file, [
            "global _start",
            "global heap_base",
            "section .bss",
            "heap_base:",
            "    resb 1024*1024",
            "section .text",
            "_start:"
        ])

    @staticmethod
    def end(file: TextIO) -> None:
        Backend._emit_all(file, [
            "print:",
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
    def emit_instruction(file: TextIO, instruction: Instr) -> None:
        opcode, operand = instruction.opcode, instruction.operand

        assert len(InstrType) == 28, "make sure to account for all instruction types"

        if opcode is InstrType.PUSH:
            Backend._emit_all(file, [
                f"; {operand}",
                f"    push    {operand}"
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

        elif opcode is InstrType.JMP:
            Backend._emit_all(file, [
                f"; jmp",
                f"    jmp     {operand}"
            ])

        elif opcode is InstrType.PRINT:
            Backend._emit_all(file, [
                f"; print",
                f"    pop     rdi",
                f"    call    print"
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

        print(f"  [{i:04}]    {instr.opcode.name:8}{instr.operand if instr.operand is not None else ''}")


# ---------------------------------------------------------------------------------------------------------------------
# Entry Point
#

if __name__ == "__main__":
    main()
