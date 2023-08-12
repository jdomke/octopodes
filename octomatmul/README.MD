# octomatmul

## Build
### Requirements
- Intel MKL Library 
- support for openMP
- ezini
- FlexiBLAS
- BOOST
- Compiler that supports c++17

### Setup
#### 1.  Setup oneapi development environment 

on Linux or macOS, run 
```bash
source <install-dir>/setvars.sh
```

on Windows, run 
```
 <install-dir>\setvars.bat
```
or
```powershell
#on powershell
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
```
#### 2. Configure Makefile
Modify the first two lines of the Makefile to where you installed the Boost library and ezini. For instance, it could be
```makefile
BOOST := ${HOME}/boost_1_82_0/
EZINI := ${HOME}/ezini/
```

#### 3. Run make
```bash
make
```

# Run
Modify /config/config.ini to configurate the underlying structure of the Octopode.

Run the octopode with ./octopode for results with the default FLEXIBLAS backend or FLEXIBLAS=(BLAS_BACKEND) ./octopode to test the octopode with a select BLAS backend. 


# Known Issues
Result verification is inaccurate on larger size matrix multiplications.

