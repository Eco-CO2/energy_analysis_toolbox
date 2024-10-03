## What does this MR do?

<!-- Briefly describe what this MR is about. -->

<!-- Link related issues. Insert the issue link or reference after the word "Closes" if merging this should automatically close it. -->

## Author's checklist

- See the [contribution guidelines](https://gitlab.ecoco2.com/EPICEE/cerebro/-/wikis/Working-with-git-and-Gitlab)
- Keep documentation up to date :
  - See the documentation guidelines [here](https://gitlab.ecoco2.com/EPICEE/cerebro/-/wikis/Coding-style)
  - [ ] Document all implemented functions
  - [ ] Update doc for of all modified functions :
    - parameters
    - returned values
    - errors
    - algo
    - examples of use if needed
  - [ ] Add/update rst files in the doc folder to keep html doc up to date
- [ ] Implement dedicated unittests
  - list limit cases
  - check these cases in the tests
- [ ] Update the [example files](/examples) if necessary
- [ ] Run the tests in the following way :
  - **Install** the module (in the development environment)
  - Open a new terminal and load the production environment
  - Run the tests in the production environment using the command in the [documentation](http://epicee.gitlab-pages.ecoco2.com/wattdf)
- [ ] Update the [CHANGELOG](CHANGELOG.md) according to the work done in this MR
- [ ] Update the version in [`__init__`](/wattdf/__init__.py) according to the modifications



## Review guidelines

Some basic elements which should be checked :
- [ ] Is the issue motivating the development explained enough so that one could
  understand in the future why this development happened and why this solution
  was chosen ?
- [ ] Are the parameters and returned values of functions consistent with the doc ?
- [ ] Are the raised errors explained in the doc ?
- [ ] Are there tests for the implemented functionalities ?
- [ ] Do the functions and tests cover the limit cases ?
- [ ] In case of a bugfix, is there a test reproducing the bug's situation in order
  to ensure non-regression?

All reviewers can help ensure accuracy, clarity, completeness, and adherence to the [Documentation and Style Guide](https://gitlab.ecoco2.com/EPICEE/cerebro/-/wikis/Coding-style).

The consistency of the interface could be a matter of interest as well.

Advices regarding implementation efficiency are most welcome.
