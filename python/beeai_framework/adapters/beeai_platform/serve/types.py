from typing import Annotated, TypedDict

try:
    import a2a.types as a2a_types
    import beeai_sdk.a2a.extensions as beeai_extensions
    import beeai_sdk.a2a.types as beeai_types
    import beeai_sdk.server.agent as beeai_agent
    import beeai_sdk.server.context as beeai_context

    from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


class BaseBeeAIPlatformExtensions(TypedDict, total=True):
    form: Annotated[
        beeai_extensions.FormExtensionServer,
        beeai_extensions.FormExtensionSpec(params=None),
    ]
    trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()]
