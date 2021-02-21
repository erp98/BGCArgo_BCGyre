# BGCArgo_BCGyre

A new an improved version of BGCArgoPython with better and more organized code :)

## Things To Do
- [X] Sort Argo floats by geographic regions
- [X] Calculate air-sea oxygen flux
- [ ] Make air-sea flux time series for different float types
- [ ] Sort floats by profile more closely to specific geographic regions
- [ ] Interpolate data to same pressure levels
- [ ] Look at oxygen over time in different regions along isopycnals?

## Python Scripts
- ***GetArgoFiles.py***: Uses synthetic profile index to get BGC Argo floats in a certain geographic region (0-80ºW, 40-80ºN)
  - Outputs DacWMO, WMO, Index text files, and floats sorted by dac (use for downloading floats from ftp site)
  - Things to change:
    - [ ] Add section that can sort by float sensor type (right now does all BGC Argo but should specify oxygen etc.)
- ***SortFloatsByType.py***: Visually inspect float trajectories and categorize them by 'type'
  - Types: Boundary current, Labrador Sea, Irminger Sea, Other, or N/A
  - Output: Sorted_WMO_<TYPE>Float.txt and Sorted_DACWMO_<TYPE>Float.txt
- ***BGCArgoGasFlux.py***: Calculates the air-sea oxygen gas flux for BGC Argo floats
  - Interpolates Argo float data to match ERA5 data timesteps, uses nearest neighbor to get ERA5 data and calculate air-sea flux at each point
  - Input: Sorted_DACWMO_<TYPE>Float.txt files and ERA5 wind and SLP data
  - Output: Figures of surface variables (temperature, salinity, oxygen) float trajectory, and air-sea oxygen flux
  - Things To Do:
    - [ ] Check code values with MATLAB ones
    - [ ] Debug section interpolation code --> no oxygen ?
    - [ ] Maybe also just have raw data or interpolated to pressure (no time interp --> make this a new script)
    - [X] Change density to sigma-thera
    - [X] Change units for air-sea flux to mol/m^2-s
    - [X] Debug section interpolation code (pres x date x parameter)
    - [X] Position check --> remove data not in the correct lat-lon range
    - [X] Quality control check --> remove data that do not have certain QC flag
    - [X] Debug code to run through all floats
    - [X] Debug L13 code and re-run code with both (L13 and N16) parameterizations
    - [X] Make float trajectory map with continents and zoomed in
    - [X] Average time-series to make data smoother
  - Notes:
    - Averages values in 5 +/- 3 dbar range
    - If there are no measurements of T, S, or O, gas flux is not calculated
    - Times:
      - 2: Boundary Current
- ***GasFluxTimeSeries.py***:
  - Things To Do:
    - [X] Make monthly averages!
    - [X] Add other fluxes (L13 and moving averages)
    - [ ] Error propagation: Calculate the standard deviation for each point w/ rolling mean
- ***MakeBoundaryCurrent.py***: Try and classify profiles as boundary current or gyre floats
  - Notes:
    - There is not statistically significant difference in float speed between the gyre and the boudary current :( However, I can define a boundary current and gyre shape with some overlap
  - Things To Do:
    - [X] Work on gyre shape --> convert convex hull to shapely polygon
    - [ ] Evaluate shape performance using other float trajectories
    - [ ] "Optimize" the fit by adjusting boundary current thickness and gyre radius
    - [ ] Try machine learning clustering techniques? Or something similar? Could be improved fit but also need to do more research
- ***RandomFxns.py***: A collection of random functions
  - DetermineAdjusted: determines if adjusted/not-adjust Argo float data is used; if adjusted data is present, these data are used
    - Output: adjusted_flag; if flag = 1 adjusted data were used
  - Other functions to add
    - [X] PositionCheck
    - [X] QCheck
    - [ ] Calculate MLD

## Questions & Other Notes & More
- Things To Do Before 2/4:
  - [X] For GasFluxTimeSeries:
    - [X] Add other fluxes (L13 and moving averages)
    - [X] Do monthly averages for the different time series
  - [ ] For MakeBoundaryCurrent:
    - [ ] Figure out why optimization is so effing slow
    - [ ] Maybe try ML or just run with one for now

- go back and check calibration of the one deep float
- check signs for parameterizations
- go-ship data check
- plotly interactive sections
