from src.typestack import TypeStack
import src.typs as ty
import src.st as st


class Checker:
    def __init__(self, definitions: list[st.Definition]) -> None:
        self._defs_to_check = definitions
        self._errors = []

    def check(self, builtin_env: dict[str, ty.Type] | None = None) -> bool:
        return all(self.check_definition(defn, builtin_env or {}) for defn in self._defs_to_check)

    def report_errors(self) -> None:
        for error in reversed(self._errors):
            print(error)

    def check_definition(self, defn: st.Definition, env: dict[str, ty.Type]) -> bool:
        if isinstance(defn, st.WordDef):
            return self.check_word_definition(defn, env)

        else:
            self._errors.append(f"{defn.location} Unhandled definition type.")
            return False

    def check_word_definition(self, word: st.WordDef, env: dict[str, ty.Type]) -> bool:
        stack = TypeStack()
        if not self.check_expression(word.body, env, stack):
            self._errors.append(f"{word.location} Failed to type WORD '{word.name}'.")
            return False

        ins, outs = stack.as_effect()
        env[word.name] = ty.Word(ins, outs)
        return True

    def check_expression(self, expr: st.Expression, env: dict[str, ty.Type], stack: TypeStack) -> bool:
        for element in expr.elements:
            if isinstance(element, st.Int):
                stack.push(ty.Int())

            elif isinstance(element, st.String):
                stack.push(ty.Str())

            elif isinstance(element, st.Char):
                stack.push(ty.Char())

            elif isinstance(element, st.Word):
                typ = env[element.name]
                if isinstance(typ, ty.Word):
                    try:
                        stack.apply_effect(typ.ins, typ.outs)
                        continue
                    except TypeError as e:
                        self._errors.append(f"{element.location} {e}.")
                        return False

                stack.push(typ)

            elif isinstance(element, st.Statement):
                if not self.check_statement(element, env, stack):
                    return False

        return True

    def check_statement(self, stmt: st.Statement, env: dict[str, ty.Type], stack: TypeStack) -> bool:
        if isinstance(stmt, st.Locals):
            child_env = env.copy()

            for name in reversed(stmt.names):
                child_env[name] = stack.pop_one()

            if not self.check_expression(stmt.body, child_env, stack):
                self._errors.append(f"{stmt.location} Could not type local binding.")
                return False

        else:
            self._errors.append(f"{stmt.location} Unhandled statement type.")
            return False

        return True
