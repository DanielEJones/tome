from dataclasses import dataclass


class Type:
    ...


@dataclass
class Int(Type):
    ...


@dataclass
class String(Type):
    ...


@dataclass
class Char(Type):
    ...


@dataclass
class Named(Type):
    name: str


@dataclass
class TVar(Type):
    name: str


def substitute(t: Type, subs: dict[str, Type]):
    if isinstance(t, TVar) and t.name in subs:
        # Continue substituting until we hit the most concrete
        # type that we can at this stage
        return substitute(subs[t.name], subs)

    # No substitutions to be made
    return t


def unify(a: Type, b: Type, subs: dict[str, Type]) -> tuple[bool, dict[str, Type]]:
    # We perform the unification on the most concrete version of the types
    a = substitute(a, subs)
    b = substitute(b, subs)

    # If a and b are the same types, which is not a variable type,
    # then a and b unify and the substitution table remains as is
    if type(a) == type(b) != TVar:
        return True, subs

    # If at least one is a variable type, the two types unify, and
    # we update the substitution table to reflect the unification
    if isinstance(a, TVar):
        subs[a.name] = b
        return True, subs

    if isinstance(b, TVar):
        subs[b.name] = a
        return True, subs

    # Otherwise, we cannot unify the types
    return False, subs


class Stack:
    def __init__(self) -> None:
        self._stack: list[Type] = []

    def top_n(self, n: int) -> list[Type]:
        if n == 0: return []
        return self._stack[-n:]

    def pop_n(self, n: int) -> None:
        if n == 0: return
        self._stack = self._stack[:-n]

    def has_at_least(self, n: int) -> bool:
        return n <= len(self._stack)

    def print(self) -> None:
        print(self._stack)

    def values(self) -> list[Type]:
        return self._stack

    def apply_effect(self, takes: list[Type], gives: list[Type]) -> tuple[bool, dict[str, Type]]:
        number_of_inputs = len(takes)

        # If there is not enough types on the stack, we cannot apply
        # we cannot apply the effect without inference
        if not self.has_at_least(len(takes)):
            return False, {}

        # Because we take values from the top of the stack first, we
        # need to unify in reverse order for the messages to make sense
        types_to_unify = zip(reversed(takes), reversed(self.top_n(number_of_inputs)))
        sub_table = {}

        for expected, got in types_to_unify:
            unified, sub_table = unify(expected, got, sub_table)
            if not unified:
                return False, sub_table

        # We managed to unify everything, so pop the old values of the
        # stack and push the result of the effect, making substitutions
        # as necessary based on the unification
        self.pop_n(number_of_inputs)
        self._stack.extend([substitute(t, sub_table) for t in gives])
        return True, sub_table


class InfiniteStack(Stack):
    def __init__(self):
        super().__init__()
        self._current_label = 0

    def _ensure_at_least(self, n: int) -> None:
        while len(self._stack) < n:
            self._stack.insert(0, self._fresh_label())

    def _fresh_label(self) -> TVar:
        label = f"_t{self._current_label}"
        self._current_label += 1
        return TVar(label)

    def top_n(self, n: int) -> list[Type]:
        self._ensure_at_least(n)
        return super().top_n(n)

    def pop_n(self, n: int) -> None:
        self._ensure_at_least(n)
        super().pop_n(n)

    def has_at_least(self, n: int) -> bool:
        return True

    def max_label(self) -> int:
        return self._current_label


StackEffect = tuple[list[Type], list[Type]]


def compose_effects(effects: list[StackEffect]) -> StackEffect | None:
    stack = InfiniteStack()
    sub_table = {}

    # The type of a sequence of effects is roughly the inputs
    # of the first effect and the outputs of the last, so to
    # type this, run through the effects applying them in order
    for takes, returns in effects:
        res, subs = stack.apply_effect(takes, returns)
        if not res: return None
        sub_table.update(subs)

    # Each label generated must feature in the inferred type of the
    # function, because they are only generated when used
    labels_used = [f"_t{n}" for n in range(stack.max_label())]

    # Find the most concrete substitution for each label generated
    args = [substitute(sub_table.get(label, TVar(label)), sub_table) for label in labels_used]
    args.reverse()

    # Any types left on the stack are types to be returned, so we
    # should find their most concrete substitution to be useful
    rets = [substitute(ty, sub_table) for ty in stack.values()]

    return args, rets


def main():
    res = compose_effects([
        ([], [Int()]),
        ([], [Int()]),
        ([Int(), Int()], [Int()]),
        ([Int()], [String()]),
        ([String()], [])
    ])

    if res is None:
        print("Type Checking Failed.")
        return

    args, rets = res
    print(f"{args=}, {rets=}")


if __name__ == "__main__":
    main()
