from src.typestack import TypeStack
import src.typs as ty
import src.st as st


def check_expression(expr: st.Expression, env: dict[str, ty.Type], stack: TypeStack) -> None:
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
                stack.apply_effect(typ.ins, typ.outs)
                continue
            stack.push(typ)

        elif isinstance(element, st.Statement):
            check_statement(element, env, stack)


def check_statement(stmt: st.Statement, env: dict[str, ty.Type], stack: TypeStack) -> None:
    if isinstance(stmt, st.Locals):
        child_env = env.copy()

        for name in reversed(stmt.names):
            child_env[name] = stack.pop_one()

        check_expression(stmt.body, child_env, stack)

    else:
        print("This is not implemented.")
        exit(1)


def check_word_definition(word: st.WordDef, env: dict[str, ty.Type]) -> None:
    stack = TypeStack()
    check_expression(word.body, env, stack)

    ins, outs = stack.as_effect()
    print(ins, outs)
    env[word.name] = ty.Word(ins, outs)
