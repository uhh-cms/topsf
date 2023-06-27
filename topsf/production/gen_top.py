# coding: utf-8

"""
Column producers related to gen-level top quark.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={"nGenPart", "GenPart.genPartIdxMother", "GenPart.pdgId", "GenPart.statusFlags"},
    produces={"GenTopDecay.products"},
)
def gen_top_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [
            # event 1
            [
                # top 1
                [t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)],
                # top 2
                [...],
            ],
            # event 2
            ...
        ]
    """
    # find hard top quarks
    abs_id = abs(events.GenPart.pdgId)
    t = events.GenPart[abs_id == 6]
    t = t[t.hasFlags("isHardProcess")]
    t = t[~ak.is_none(t, axis=1)]

    # distinct top quark children (b's and W's)
    t_children = t.distinctChildrenDeep[t.distinctChildrenDeep.hasFlags("isHardProcess")]

    # get b's (or very rarely other down-like quark from prompt top decay)
    down_like_quark = (
        (abs(t_children.pdgId) == 5) |
        (abs(t_children.pdgId) == 3) |
        (abs(t_children.pdgId) == 1)
    )
    b = t_children[down_like_quark][:, :, 0]

    # get W's
    w = t_children[abs(t_children.pdgId) == 24][:, :, 0]

    # distinct W children
    w_children = w.distinctChildrenDeep[w.distinctChildrenDeep.hasFlags("isHardProcess")]

    # reorder the first two W children (leptons or quarks) so that the charged lepton / down-type
    # quark is listed first (they have an odd pdgId)
    w_children_firsttwo = w_children[:, :, :2]
    w_children_firsttwo = w_children_firsttwo[(w_children_firsttwo.pdgId % 2 == 0) * 1]
    w_children_rest = w_children[:, :, 2:]

    # concatenate to create the structure to return
    groups = ak.concatenate(
        [
            t[:, :, None],
            b[:, :, None],
            w[:, :, None],
            w_children_firsttwo,
            w_children_rest,
        ],
        axis=2,
    )

    # save the column
    events = set_ak_column(events, "GenTopDecay.products", groups)

    return events


@gen_top_decay_products.skip
def gen_top_decay_products_skip(self: Producer) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing top
    quarks in the first place.
    """
    # never skip when there is not dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")


@producer(
    uses={
        gen_top_decay_products,
    },
    produces={
        gen_top_decay_products,
        "GenTopDecay.n_had",
    },
)
def gen_top_decay(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produce gen top decay information:
      * number of hadronically decaying top quarks
    """

    n_had = 0

    # get decay products of top quark
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_decay_products](events, **kwargs)

        q1_or_l = events.GenTopDecay.products[:, :, 3]  # light quark 1 / lepton
        q2_or_n = events.GenTopDecay.products[:, :, 4]  # light quark 2 / neutrino

        n_had = (
            ak.num(abs(q1_or_l.pdgId) <= 5, axis=1) +
            ak.num(abs(q2_or_n.pdgId) <= 5, axis=1)
        )

    # write out columns
    events = set_ak_column(events, "GenTopDecay.n_had", n_had)

    return events
