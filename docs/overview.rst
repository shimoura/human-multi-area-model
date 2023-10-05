Overview
========

von Economo & HCP model
-----------------------

Ingredients
+++++++++++

* von Economo & Koskinas cytoarchitecture (human)
* Human Connectome Project DTI data (human)
* FLN data from Markov et al. (macaque)
* SLN prediction based on cytoarchitecture from MAM (macaque)
* Binzegger data for synapse to cellbody location (cat)
* Spatial connectivity profile (TODO)

Microcircuit-level connectivity
+++++++++++++++++++++++++++++++

.. figure::  ../notes/graph/microcircuit_connectivity_graph.svg
    :align: center

Splitting type I and II
+++++++++++++++++++++++

Given a spatial connection probability :math:`p(x_1, x_2)` between two neurons at :math:`x_1`
and :math:`x_2`, the probability that a connection from a neuron inside (outside)
of the microcircuit to a neuron inside the microcircuit exists is given by

.. math::
   P_{\mathrm{in/out}} = \int_{\mathrm{in/out}}dV_1\, \int_{\mathrm{in}}dV_2\, p(x_1, x_2).

Then, the relative number of type I/II synapses is simply

.. math::
   N_{\mathrm{syn}, A}^\mathrm{I/II} = \frac{P_\mathrm{in/out}}{P_\mathrm{in} + P_\mathrm{out}}N_{\mathrm{syn}, A}^\mathrm{I+II}.

For the spatial connection probability, we assume a spatially homogeneous three dimensional exponential distribution with
a decay length of :math:`160 \mu m` (Packer & Yuste 2011, Perin et al. 2011).

Since we model the microcolumn as a cylinder, the natural choice are cylindrical coordinates
for which :math:`dV\, = r dr\,d\phi\,dz\,` and
:math:`|x_1 - x_2|^2 = r_1^2 - 2r_1r_2\cos(\phi_2-\phi_1) + r_2^2 + (z_2-z_1)^2`.
Moreover, we can use :math:`\int_0^a dx_1\, \int_0^a dx_2\, f(x_2 - x_1) = \int_0^a dy\, (a-y)[f(y) + f(-y)]`
to arrive at

.. math::
   P_\mathrm{in} &= \int_{\mathrm{in}}dV_1\, \int_{\mathrm{in}}dV_2\, p(|x_1 - x_2|) \\
   &= 4 \int_0^R dr_1\, \int_0^R dr_2\, \int_0^{2\pi} d\phi\, \int_0^h dz\, r_1 r_2 (2\pi - \phi) (h - z) p(r_1, r_2, \phi, z)

where :math:`\phi=\phi_2-\phi_1` and :math:`z=z_2-z_1`. For :math:`P_\mathrm{out}`, we simply have
to adjust the limits of the :math:`r_1` integration.

Intra-microcircuit connectivity (I)
+++++++++++++++++++++++++++++++++++

We assume the connection probabilities to be proportional to the connection probabilities of the PD model.
Using this, we can assign synapses to the respective populations:

.. math::
   N_{\mathrm{syn}, A}^\mathrm{I}(x, y) = \frac{N_{x, A} p^{\mathrm{PD}}(x, y) N_{y, A}}{\sum_{x, y} N_{x, A} p^{\mathrm{PD}}(x, y) N_{y, A}} N_{\mathrm{syn}, A}^\mathrm{I}.

Here, :math:`y` is the source and :math:`x` the target population and :math:`N_x` the number of neurons in
population :math:`x`.

Extra-microcircuit connectivity (II)
++++++++++++++++++++++++++++++++++++

Here, we follow the MAM and assume the indegrees to be proportional to the indegrees of the PD model.
Using this, we can assign external synapses to the respective populations:

.. math::
   N_{\mathrm{syn}, A}^\mathrm{II}(x) = \frac{N_{x, A} K_{\mathrm{ext}}^{\mathrm{PD}}(x)}{\sum_{x} N_{x, A} K_{\mathrm{ext}}^{\mathrm{PD}}(x)} N_{\mathrm{syn}, A}^\mathrm{II}.

Here, :math:`x` is the target population and :math:`N_x` the number of neurons in
population :math:`x`.

Cortico-cortical connectivity (III)
+++++++++++++++++++++++++++++++++++

.. figure::  ../notes/graph/huvi_cc.svg
    :align: center

The SLN is based on the cytoarchitecture using the fit

.. math::
   SLN(A, B) = \frac{1}{2}\left\{ 1 + \mathrm{erf}\, \left[ \tfrac{1}{\sqrt{2}} (a_0 + a_1 \log(\rho_A / \rho_B)) \right] \right\}

based on the Markov data.
