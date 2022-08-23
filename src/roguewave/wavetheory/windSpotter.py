"""
This file is part of pysofar: A client for interfacing with Sofar Oceans Spotter API

Contents: Wind Estimater

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import typing
import numpy

def U10( E , f , a1=None , b1=None  ,fmin=-1., fmax=.5  ,Npower=4        , I=2.5 ,
               beta=0.012, kappa=0.4,alpha=0.012, grav=9.81, numberOfBins=20)->typing.Tuple[float,float,float]:
    #
    # =========================================================================
    # Required Input
    # =========================================================================    
    # 
    # f              :: frequencies (in Hz)
    # E              :: Variance densities (in m^2 / Hz )
    #
    # =========================================================================
    # Output
    # =========================================================================    
    #
    # U10            :: in m/s
    # Direction      :: in degrees clockwise from North (where wind is *coming from)
    #
    # =========================================================================
    # Named Keywords (parameters to inversion algorithm)
    # =========================================================================
    # Npower = 4     :: exponent for the fitted f^-N spectral tail
    # I      = 2.5   :: Philips Directional Constant
    # beta   = 0.012 :: Equilibrium Constant
    # Kapppa = 0.4   :: Von Karman constant
    # Alpha  = 0.012 :: Constant in estimating z0 from u* in Charnock relation
    # grav   = 9.81  :: Gravitational acceleration
    #
    # =========================================================================
    # Algorithm
    # =========================================================================    
    #
    # 1) Find the part of the spectrum that best fits a f^-4 shape
    # 2) Estimate the Phillips equilibrium level "Emean" over that range
    # 3) Use Emean to estimate Wind speed (using Charnock and LogLaw)   
    # 4) Calculate mean direction over equilibrium range
    
    
    
    # If fmin smaller than 0; use fraction of peak frequency as minimum
    if fmin < 0.:
        #
        iMin = numpy.argmax( E )
        fmin = numpy.abs(fmin) * f[iMin]
        #
    #
    
    #
    # Find the equilibrium range; output is a list of indices of the spectrum array that make up the tail
    #
    equilibriumRange = findEquilibriumRange( E , f ,  fmin=fmin , fmax=fmax ,  Npower=Npower, numberOfBins = numberOfBins)    
    #
    # Caclulate the weighted mean over the tail
    
    Emean            = 8. * numpy.pi**3 * numpy.mean( E[equilibriumRange] * f[equilibriumRange]**Npower )
        
    # Get friction velocity from spectrum
    Ustar    = Emean / grav / I / beta / 4

    # Find z0 from Charnock Relation
    z0       = alpha * Ustar**2 / grav
        
    # Get the wind speed at U10 from loglaw
    U  = Ustar / kappa * numpy.log( 10. / z0 )        
    
    #
    if a1 is not None:
        #
        # Calculate the mean directional moments over the equilibrium range.
        #
        a1m = numpy.mean( a1[equilibriumRange] )
        b1m = numpy.mean( b1[equilibriumRange] )   
        #
        # Convert directions to where the wind is coming from, measured positive clockwise from North
        dir = (270. - 180. / numpy.pi * numpy.arctan2( b1m , a1m )) % 360
                #
    else:
        #
        dir = None
        #
    #
    return( U , dir, Ustar )
    #
#enddef

def findEquilibriumRange( E , f ,  fmin=0.0293 , fmax=1.25 ,  Npower=4, numberOfBins = 20):
    #
    # fmin        :: minimum frequency considered (default is Spotter minimum)
    # fmax        :: maximum frequency considered (default is Spotter Maximum)
    import numpy

    #Get power spectrum
    E = numpy.squeeze(  E * f ** Npower )
    
    #Find fmin/fmax
    iMin = numpy.argmin( numpy.abs( f - fmin ) ,axis= -1 )
    iMax = numpy.argmin( numpy.abs( f - fmax ) ,axis= -1 )
    nf   = len(f)
    
    numberOfFrequencies   = E.shape[-1]
    iMax                  = iMax + 1 - numberOfBins
    iMax                  = numpy.max( (iMin + 1, iMax) )
    iMax                  = numpy.min( (iMax , nf-numberOfBins) )
    
    iCounter = 0
    Variance = numpy.zeros( iMax-iMin ) + 1.e10
    #
    # Calculate the variance with respect to a running average mean of numberOfBins
    #
    for iFreq in range( iMin , iMax ):
        #        
        #Ensure we do not go out of bounds
        iiu = numpy.min( (iFreq+numberOfBins, nf )  )
        
        #Ensure there are no 0 contributions (essentially no data)
        msk = E[iFreq:iiu] > 0.
        if (numpy.sum(msk) == numberOfBins):
            M                  = numpy.mean(  E[iFreq:iiu]         )
            Variance[iCounter] = numpy.mean( (E[iFreq:iiu] - M)**2 ) / ( M**2 )
            
        iCounter = iCounter + 1        
        #
    #
    iMinVariance = numpy.argmin( Variance )
    iiu = numpy.min( (iMinVariance + iMin  + numberOfBins, nf ) )
    #
    return ( numpy.arange( iMinVariance + iMin, iiu ) )
    #
#enddef