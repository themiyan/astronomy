def ivarsmooth(flux, ivar, width, weight=None, profile=None):
   """Generates an inverse variance weighted 1d spectrum. 
   flux- science spectrum to be smoothed
   ivar- inverse variance spectrum of the science spectrum
   width- the number of pixels to be smoothed
   weight- weight to be applied per pixel (default=1)
   profile- profile to be applied (default=1)
   
   NOTE: This is expected to give  similar results as ivarsmooth.pro from ucolick: http://tinyurl.com/nm5udej
   
   @themiyan 16/06/2015
   """
    import scipy.ndimage.filters as fil
    
    if weight  == None:
        weight=1
    if profile == None:
        profile=1
    width = np.ones((1,width), dtype=float)[0]

    numerator             = flux * ivar * weight * profile 

    convolved_numerator   = fil.convolve1d(numerator , width, axis=-1, mode='reflect', cval=0.0, origin=0 )
    
    denominator           = ivar * weight * profile
    
    convolved_denominator = fil.convolve1d( denominator, width, axis=-1, mode='reflect', cval=0.0, origin=0 )

    smoothed_flux         = convolved_numerator / convolved_denominator 

    smoothed_ivar         = convolved_denominator
    
    return smoothed_flux, smoothed_ivar
