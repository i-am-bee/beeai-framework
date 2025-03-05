from beeai_framework.errors import FrameworkError


def get_user() -> None:
    raise FrameworkError("User not found")


try:
    get_user()
except Exception as e:
    err = FrameworkError.ensure(e)
    print(err.get_cause())
    print(err.explain())
