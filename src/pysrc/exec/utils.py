from typing import Any

def get_input(prompt: str, default: Any, vals: list[Any]) -> Any:
    while True:
        inp = input(prompt)

        if inp in vals:
            return inp
        elif inp == "":
            return default
        else:
            print("incorrect format, please try again")