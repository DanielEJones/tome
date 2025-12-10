from src.typs import Type, TVar


class TypeStack:
    def __init__(self) -> None:
        self._stack = []
        self._subs = {}
        self._arg_count = 0
        self._var_count = 0

    def push(self, *ts: Type) -> None:
        self._stack.extend(ts)

    def pop_one(self) -> Type:
        if len(self._stack) < 1:
            return self._fresh_arg()
        return self._stack.pop()

    def pop(self, n: int) -> list[Type]:
        while len(self._stack) < n:
            self._stack.insert(0, self._fresh_arg())
        self._stack, values = split_list_at(len(self._stack) - n, self._stack)
        return values

    def apply_effect(self, ins: list[Type], outs: list[Type]) -> None:
        subs = {}
        ins = self._normalize_vars(ins, subs)
        outs = self._normalize_vars(outs, subs)

        types_to_unify = zip(reversed(ins), reversed(self.pop(len(ins))))
        for expected, got in types_to_unify:
            if not self.unify(expected, got):
                raise TypeError(f"Could not unify {self.substitute(expected)} and {self.substitute(got)}.")

        self.push(*outs)

    def as_effect(self) -> tuple[list[Type], list[Type]]:
        ins = [self.substitute(TVar(f"A{n}")) for n in reversed(range(self._arg_count))]
        outs = [self.substitute(v) for v in self._stack]
        return ins, outs

    def unify(self, a: Type, b: Type) -> bool:
        a = self.substitute(a)
        b = self.substitute(b)

        if a == b and not isinstance(a, TVar):
            return True

        if isinstance(a, TVar):
            self._subs[a.name] = b
            return True

        if isinstance(b, TVar):
            self._subs[b.name] = a
            return True

        return False

    def substitute(self, t: Type) -> Type:
        if isinstance(t, TVar) and (n := t.name) in self._subs:
            return self.substitute(self._subs[n])
        return t

    def _fresh_arg(self) -> TVar:
        label = f"A{self._arg_count}"
        self._arg_count += 1
        return TVar(label)

    def _fresh_var(self) -> TVar:
        label = f"T{self._var_count}"
        self._var_count += 1
        return TVar(label)

    def _normalize_vars(self, ts: list[Type], subs: dict[str, Type] | None = None) -> list[Type]:
        sub_table = {} if subs is None else subs
        result = []

        for t in ts:
            if isinstance(t, TVar) and t.name in sub_table:
                result.append(sub_table[t.name])
            elif isinstance(t, TVar):
                fresh = self._fresh_var()
                sub_table[t.name] = fresh
                result.append(fresh)
            else:
                result.append(t)

        return result


def split_list_at(n: int, l: list) -> tuple[list, list]:
    return l[:n], l[n:]
