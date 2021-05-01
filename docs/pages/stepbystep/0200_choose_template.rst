Choose a Model Template
=======================

Model templates provide default model parameters
and allow you to customize them in a more organized way.
Greykite offers various templates for the two forecasting
models: Silverkite and Prophet.

.. note::

  If possible, try all the templates with default parameters. Tune a few of the main parameters,
  then focus on the one that looks most promising for your dataset.

  You can use the same forecaster
  (:class:`~greykite.framework.templates.forecaster.Forecaster`)
  to run different templates and compare the results.

  That said, there are some clear differences between templates, and this cheatsheet
  can help you find the right one for your problem.

See valid ``model_template`` names in
`~greykite.framework.templates.model_templates.ModelTemplateEnum`.
The model templates can be classified into three categories:

* The ``SILVERKITE`` category of model templates provides a high-level interface to the
  Silverkite model. This category of templates includes configurations that are
  tailored to various forecast horizons, data frequencies, and data characteristics.
  It is easy to customize these configurations to try different options.
  The model template names are strings starting with ``"SILVERKITE"`` or instances of
  `~greykite.framework.templates.simple_silverkite_template_config.SimpleSilverkiteTemplateOptions`.
  The class that applies these templates is
  `~greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate`.
* The ``"SK"`` model template is a low-level interface to the Silverkite model.
  This model template allows you to change lower-level parameters in Silverkite
  and is intended for more advanced users.
  The class that applies this template is
  `~greykite.framework.templates.silverkite_template.SilverkiteTemplate`.
* The ``"PROPHET"`` model template is used for the Prophet model.
  The class that applies this template is
  `~greykite.framework.templates.prophet_template.ProphetTemplate`.

A detailed tutorial and listing of available model templates can be found at
:doc:`/gallery/tutorials/0200_templates`.