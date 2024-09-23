.. Sphinx Docs example documentation master file, created by
   sphinx-quickstart on Thu Feb 16 10:23:44 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to energy_toolbox documentation!
===============================================

.. grid:: 1 2 2 2

   .. grid-item-card:: Getting started
      :text-align: center

      .. image:: /_static/index_getting_started.svg
         :width: 100px

      Getting started
      ^^^^^^^^^^^^^^^
      New to |et| ? Check out the getting started guides. They contain an
      introduction to |et|'s main concepts and links to additional tutorials.
      +++

      .. button-ref:: getting_started/index
         :expand:
         :color: secondary
         :click-parent:


   .. grid-item-card:: User guide
      :text-align: center

      .. image:: /_static/index_user_guide.svg
         :width: 100px

      User guide
      ^^^^^^^^^^
      The user guide provides in-depth information on the
      key concepts of |et| with useful background information and explanation.
      +++

      .. button-ref:: user_guide/index
         :expand:
         :color: secondary
         :click-parent:

   .. grid-item-card:: API reference
      :text-align: center

      .. image:: /_static/index_api.svg
         :width: 100px

      API reference
      ^^^^^^^^^^^^^
      The reference guide contains a detailed description of
      the |et| API. The reference describes how the methods work and which parameters can
      be used. It assumes that you have an understanding of the key concepts.
      +++

      .. button-ref:: sources/modules
         :expand:
         :color: secondary
         :click-parent:

   .. grid-item-card:: Developer guide
      :text-align: center

      .. image:: /_static/index_contribute.svg
         :width: 100px

      Developer guide
      ^^^^^^^^^^^^^^^
      Saw a typo in the documentation? Want to improve
      existing functionalities? The contributing guidelines will guide
      you through the process of improving |et|.
      +++

      .. button-ref:: developer_guide/contributing
         :expand:
         :color: secondary
         :click-parent:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   getting_started/index
   user_guide/index
   sources/modules
   developer_guide/contributing