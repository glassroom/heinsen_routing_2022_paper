# coding: utf-8

class SequentialCreditAssignmentsHandler(object):
    """
    Uses PyTorch forward hooks to store the credit assignments computed by sequential
    routings, and computes end-to-end credit assignments following the instructions
    in Appendix A of "An Algorithm for Routing Vectors in Sequences" (Heinsen, 2022).
    """

    def __init__(self, eps=1e-5):
        self.eps = eps
        self.hooks = []
        self.clear()

    def clear(self):
        self.phis = []

    def _forward_hook(self, routing, inp, out):
        self.phis.append(out['phi'].detach())
        return out['x_out']

    def add_forward_hook(self, routing):
        self.hooks.append(routing.register_forward_hook(self._forward_hook))

    def end_to_end_prod(self):
        prod = None
        for phi in self.phis:
            prod = phi if prod is None else prod @ phi  # chain of matrix products
            prod /= (prod.std() + self.eps)             # scale chain of products
        return prod

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []