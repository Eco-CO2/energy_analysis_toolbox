# energy_toolbox

`ct` is a library dedicated to data manipulation and processing for common tasks
encountered in the daily life of energy/data engineers here at Eco CO2.

See the [documentation](http://recherche.gitlab-pages.ecoco2.com/energy_toolbox/html/)
for information about how to get started with the lib, the API reference, user-guide
and contributing guidelines.

## What to expect in this Toolbox ?

### The Moto of CT

- No dependances to other Eco CO2 libraries (`cerebro`, `nb_toolbox`, `data_connector`, etc.)
- Not related to a specify database, service, or product
- Contains functionalities which have been validated and tested. MOre experimental
  ones are to be found in `mnb_toolbox`.

### What you can find

- Generic Tools to process deferent types of data (power, energy, temperature, etc.)
    - Resamplers that can deal with missing data
    - Aggregators that compute daily profiles
    - Pre-processing that clean the data from ouliers
- Specialised tools
    - For Weather : DJU computations
    - Energy precessing, that can classify different periods or apply weather correction.
    - TBD
- Tests to verify the precision of the processing tools
- a beautifull documentation that explain **How** to use the tools, as well as
  **Why** these implementations as been choosen.