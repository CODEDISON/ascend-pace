"""F-arm recorder-removal machinery and its CPU fake-import proof.

Pure CPU: importing this file imports no framework and touches no device.

Semantics (the PACE shape of the native timing lane's ten-binding discipline):
the ten targets are snapshotted BEFORE the frozen sync-model install; after
that install (PACE hook + recorder wrappers in one RPC) each target's full
`__wrapped__` chain is recorded by code fingerprint. Restoration removes
EXACTLY ONE layer — the outermost (recorder) wrapper — so the layer the
frozen install put underneath (the PACE hook for the comm-path targets) is
retained through the timing region, and the recorder stays removed and
byte-identical afterwards. Every identity is bound by callable identity and
code fingerprint; a partial failure is terminal and mutates nothing.

`fake_import_proof` exercises the whole machine against synthetic modules,
including a two-layer chain (recorder over hook over original), double-
restore refusal and forged-target rejection. The device complement is the
one-request restoration proof (PLAN section 13.2) gated in launch.py.
"""
import hashlib
import json
from types import MethodType

NAMES = ('runner.execute_model', 'runner._dummy_run', 'context.forward', 'runner_module.forward',
         'context.selector', 'dispatcher.helper', 'all2all.dispatch', 'all2all.combine',
         'mc2.dispatch', 'mc2.combine')


class RestorationError(ValueError):
    pass


def _fingerprint(code):
    return hashlib.sha256(
        json.dumps([code.co_name, code.co_filename, code.co_firstlineno,
                    len(code.co_code)], sort_keys=True).encode()).hexdigest()


def fingerprint(fn):
    target = fn.__func__ if isinstance(fn, MethodType) else fn
    return _fingerprint(target.__code__)


def same(a, b):
    if isinstance(a, MethodType) or isinstance(b, MethodType):
        return isinstance(a, MethodType) and isinstance(b, MethodType) \
            and a.__func__ is b.__func__ and a.__self__ is b.__self__
    return a is b


def chain_of(fn):
    """Full __wrapped__ chain [outermost, ..., terminal] with fingerprints."""
    chain, seen, current = [], set(), fn
    while True:
        chain.append(current)
        if id(current) in seen:
            raise RestorationError('cyclic __wrapped__ chain')
        seen.add(id(current))
        nxt = getattr(current, '__wrapped__', None)
        if nxt is None:
            return chain
        current = nxt


def targets(worker, context, runner_module, dispatcher):
    a2a = dispatcher.TokenDispatcherWithAll2AllV
    mc2 = dispatcher.TokenDispatcherWithMC2
    return list(zip(NAMES, ((worker.model_runner, 'execute_model'), (worker.model_runner, '_dummy_run'),
                            (context, 'set_ascend_forward_context'), (runner_module, 'set_ascend_forward_context'),
                            (context, 'select_moe_comm_method'), (dispatcher, 'async_all_to_all'),
                            (a2a, 'token_dispatch'), (a2a, 'token_combine'),
                            (mc2, 'token_dispatch'), (mc2, 'token_combine'))))


class RecorderBindings:
    """Ten bindings: pre-install snapshot -> post-install chain adoption ->
    exactly-one-layer recorder removal -> verified retention of the hook."""

    def __init__(self, items):
        if tuple(name for name, _ in items) != NAMES:
            raise RestorationError('expected ten binding names')
        self.items = items
        self.phase = 'pre_install'
        self.originals = {name: getattr(obj, attr) for name, (obj, attr) in items}
        self.original_codes = {name: fingerprint(fn) for name, fn in self.originals.items()}

    def adopt_hook_layer(self):
        """After the frozen install: record each target's full wrapper chain.

        Requires every current binding to be a wrapper whose chain has at
        least two layers (recorder over something). The layer at index 1 is
        the hook-or-original layer restoration will retain.
        """
        if self.phase != 'pre_install':
            raise RestorationError('adoption state differs')
        self.chains, self.chain_codes = {}, {}
        for name, (obj, attr) in self.items:
            chain = chain_of(getattr(obj, attr))
            if len(chain) < 2:
                raise RestorationError('recorder wrapper missing on: ' + name)
            self.chains[name] = chain
            self.chain_codes[name] = [fingerprint(fn) for fn in chain]
        self.phase = 'adopted'
        return self.report()

    def check_wrapped(self):
        """No drift between adoption and removal."""
        if self.phase != 'adopted':
            raise RestorationError('adoption required before the removal check')
        for name, (obj, attr) in self.items:
            chain = chain_of(getattr(obj, attr))
            if [fingerprint(fn) for fn in chain] != self.chain_codes[name]:
                raise RestorationError('wrapper chain drifted: ' + name)
        self.phase = 'checked'
        return self.report()

    def restore(self):
        """Remove exactly the outermost (recorder) layer; retain the rest."""
        if self.phase != 'checked':
            raise RestorationError('removal requires the drift check')
        # Verify every target (identity AND fingerprint of every chain layer)
        # before mutating any: partial failure stays terminal.
        for name, (obj, attr) in self.items:
            chain = chain_of(getattr(obj, attr))
            if len(chain) != len(self.chains[name]) \
                    or any(not same(actual, saved) or fingerprint(actual) != code
                           for actual, saved, code in zip(chain, self.chains[name], self.chain_codes[name])):
                raise RestorationError('wrapper chain changed before removal: ' + name)
        self.phase = 'restoring'
        for name, (obj, attr) in self.items:
            setattr(obj, attr, self.chains[name][1])
        self.phase = 'restored'
        return self.verify()

    def verify(self):
        if self.phase != 'restored':
            raise RestorationError('the ten bindings have not been restored')
        for name, (obj, attr) in self.items:
            current = getattr(obj, attr)
            if not same(current, self.chains[name][1]) \
                    or fingerprint(current) != self.chain_codes[name][1]:
                raise RestorationError('retained layer changed: ' + name)
            if [fingerprint(fn) for fn in chain_of(current)] != self.chain_codes[name][1:]:
                raise RestorationError('retained layer chain tail differs: ' + name)
        return self.report()

    def report(self):
        return {'phase': self.phase, 'binding_count': len(self.items),
                'chain_code_sha256': {name: list(codes) for name, codes in self.chain_codes.items()},
                'removed_layers': 1,
                'scope': 'saved Python callable identities and code only; runtime files '
                         'pinned by the frozen capture installation'}


def fake_import_proof():
    """CPU proof over synthetic modules (no framework import, no device)."""
    class Code:
        def __init__(self, name, line):
            self.co_name, self.co_filename, self.co_firstlineno = name, '<fake>', line
            self.co_code = bytes([line % 251, 1])

    class Fn:
        def __init__(self, name, line):
            self.__code__ = Code(name, line)

    class Obj:
        def __init__(self, tag):
            self.token_dispatch = Fn(tag + '_dispatch', 11)
            self.token_combine = Fn(tag + '_combine', 12)

    class ModelRunner:
        def __init__(self):
            self.execute_model = Fn('execute_model', 1)
            self._dummy_run = Fn('_dummy_run', 2)

    class Worker:
        def __init__(self):
            self.model_runner = ModelRunner()

    import types
    from functools import wraps
    context = types.ModuleType('fake_context')
    runner_module = types.ModuleType('fake_runner')
    dispatcher = types.ModuleType('fake_dispatcher')
    context.set_ascend_forward_context = Fn('ctx_forward', 3)
    runner_module.set_ascend_forward_context = Fn('runner_forward', 4)
    context.select_moe_comm_method = Fn('selector', 5)
    dispatcher.async_all_to_all = Fn('helper', 6)
    dispatcher.TokenDispatcherWithAll2AllV = Obj('a2a')
    dispatcher.TokenDispatcherWithMC2 = Obj('mc2')
    worker = Worker()
    items = targets(worker, context, runner_module, dispatcher)

    def wrap(obj, attr, tag):
        inner = getattr(obj, attr)
        @wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)
        wrapper.__revision4_tag__ = tag
        setattr(obj, attr, wrapper)
        return wrapper

    bindings = RecorderBindings(items)
    # Simulated frozen install: the PACE hook replaces the two All2AllV
    # dispatcher methods (two-layer future: hook wraps the original), then
    # the recorder wraps all ten targets.
    for name in ('all2all.dispatch', 'all2all.combine'):
        obj, attr = dict(items)[name]
        wrap(obj, attr, 'pace_hook')
    wrappers = {}
    for name, (obj, attr) in items:
        wrappers[name] = wrap(obj, attr, 'recorder')
    adopted = bindings.adopt_hook_layer()
    if adopted['phase'] != 'adopted' or any(len(c) < 2 for c in bindings.chain_codes.values()):
        raise RestorationError('adoption report differs')
    if bindings.chain_codes['all2all.dispatch'].__len__() != 3 \
            or bindings.chain_codes['runner.execute_model'].__len__() != 2:
        raise RestorationError('chain depth differs from the simulated install')
    bindings.check_wrapped()
    restored = bindings.restore()
    if restored['phase'] != 'restored':
        raise RestorationError('restored report differs')
    # The recorder layer is gone; the PACE hook layer is retained exactly.
    obj, attr = dict(items)['all2all.dispatch']
    if not same(getattr(obj, attr), wrappers['all2all.dispatch'].__wrapped__):
        raise RestorationError('retained layer is not the hook layer')
    if getattr(getattr(obj, attr), '__revision4_tag__', None) != 'pace_hook':
        raise RestorationError('the recorder layer was not the removed one')
    for name, (obj, attr) in items:
        if name not in ('all2all.dispatch', 'all2all.combine') \
                and not same(getattr(obj, attr), bindings.originals[name]):
            raise RestorationError('unhooked target must return to its original: ' + name)
    try:
        bindings.restore()
        raise AssertionError('double restore must be refused')
    except RestorationError:
        pass
    # Forged wrapper target is caught before any mutation.
    bindings2 = RecorderBindings(items)
    for name, (obj, attr) in items:
        wrap(obj, attr, 'recorder2')
    bindings2.adopt_hook_layer()
    bindings2.check_wrapped()
    bindings2.chains[NAMES[0]][1] = Fn('forged', 99)
    try:
        bindings2.restore()
        raise AssertionError('forged chain must be refused')
    except RestorationError:
        pass
    return {'proof': 'cpu_fake_import_restoration',
            'binding_names': list(NAMES),
            'chain_code_sha256': bindings.report()['chain_code_sha256'],
            'removed_layers': 1,
            'hook_layer_retained': True,
            'double_restore_refused': True,
            'forged_chain_rejected': True,
            'scope': 'synthetic modules; the device complement is the one-request proof'}
