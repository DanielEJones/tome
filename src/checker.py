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
            self._errors.append(f"{word.location} Failed to typecheck '{word.name}'.")
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

            # The names have to come off in reverse order because we
            # are using pop rather than splitting off n values
            for name in reversed(stmt.names):
                child_env[name] = stack.pop_one()

            if not self.check_expression(stmt.body, child_env, stack):
                self._errors.append(f"{stmt.location} Could not type local binding.")
                return False

        elif isinstance(stmt, st.If):
            if not self.check_expression(stmt.cond, env, stack):
                self._errors.append(f"{stmt.location} Could not type condition.")
                return False

            # The condition must resolve to a state where the stack has a boolean on top
            if (cond := stack.pop_one()) != ty.Bool():
                self._errors.append(f"{stmt.location} Expected condition to produce a boolean but got {cond} instead.")
                return False

            # Both branches must type check
            then_stack = stack.clone()
            if not self.check_expression(stmt.body, env, then_stack):
                self._errors.append(f"{stmt.location} Could not type check True branch.")
                return False

            els_stack = stack.clone()
            if not self.check_expression(stmt.els, env, els_stack):
                self._errors.append(f"{stmt.location} Could not type check False branch.")
                return False

            # Both branches must have the same shape for the statement to be valid
            if not then_stack.eq(els_stack):
                self._errors.append(f"{stmt.location} Could not unify True and False branches.")
                return False

            stack.replace_with(then_stack)

        elif isinstance(stmt, st.While):
            # The condition and body have to be able to run 0, 1 or N times. This means that to be typed,
            # the condition and body both have to leave the stack in the same shape they found it, except
            # the with a boolean value on top in case of the condition

            # Check the condition
            cond_stack = stack.clone()
            if not self.check_expression(stmt.cond, env, cond_stack):
                self._errors.append(f"{stmt.location} Could not type loop condition.")
                return False

            if (cond := cond_stack.pop_one()) != ty.Bool():
                self._errors.append(f"{stmt.location} Expected condition to produce a boolean but got {cond} instead.")
                return False

            if not cond_stack.eq(stack):
                self._errors.append(f"{stmt.location} Loop condition changes the shape of the stack arbitrarily.")
                return False

            # Check the body (use a clone of the `cond_stack` to preserve any substitutions made)
            body_stack = cond_stack.clone()
            if not self.check_expression(stmt.body, env, body_stack):
                self._errors.append(f"{stmt.location} Could not type loop body.")
                return False

            if not body_stack.eq(stack):
                self._errors.append(f"{stmt.location} Loop body changes the shape of the stack arbitrarily.")
                return False

            stack.replace_with(body_stack)

        else:
            self._errors.append(f"{stmt.location} Unhandled statement type.")
            return False

        return True
