# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmselfsup into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmselfsup default
            scope. When `init_default_scope=True`, the global default scope
            will be set to `mmselfsup`, and all registries will build modules
            from mmselfsup's registry node. To understand more about the
            registry, please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import cmae.datasets  # noqa: F401,F403
    import cmae.engine  # noqa: F401,F403
    import cmae.evaluation  # noqa: F401,F403
    import cmae.models  # noqa: F401,F403
    import cmae.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('cmae')
        if never_created:
            DefaultScope.get_instance('cmae', scope_name='cmae')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'cmae':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "cmae", '
                          '`register_all_modules` will force set the current'
                          'default scope to "cmae". If this is not as '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'cmae-{datetime.datetime.now()}'
            DefaultScope.get_instance(
                new_instance_name, scope_name='cmae')
