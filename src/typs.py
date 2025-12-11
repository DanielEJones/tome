from dataclasses import dataclass


@dataclass
class Type:
    ...


class Int(Type):
    ...


class Bool(Type):
    ...


class Str(Type):
    ...


class Char(Type):
    ...


@dataclass
class Word(Type):
    ins: list[Type]
    outs: list[Type]


@dataclass
class TVar(Type):
    name: str
