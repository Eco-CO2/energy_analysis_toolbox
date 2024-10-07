![Logo](doc/_static/logo.png)

# energy_analysis_toolbox

`eat` is a library dedicated to data manipulation and processing for common tasks
encountered in the daily life of energy/data engineers here at Eco CO2.

See the [documentation](http://recherche.gitlab-pages.ecoco2.com/energy_analysis_toolbox/html/)
for information about how to get started with the lib, the API reference, user-guide
and contributing guidelines.

## What to expect in this Toolbox ?

### What you can find

- Generic Tools to process deferent types of data (power, energy, temperature, etc.)
    - Resamplers that can deal with missing data
    - Aggregators that compute daily profiles
    - Pre-processing that clean the data from ouliers
- Specialised tools
    - For Weather : DD computations
    - Energy precessing, that can classify different periods or apply weather correction.
    - TBD
- Tests to verify the precision of the processing tools
- a beautifull documentation that explain **How** to use the tools, as well as
  **Why** these implementations as been choosen.
