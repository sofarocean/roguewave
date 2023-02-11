import numpy


def least_squares_power(x,y, power):
    X = numpy.sum(x)
    Y = numpy.sum(y)
    XY = numpy.sum(x*y)
    XX = numpy.sum(x * x)
    n = len(x)

    inverse_matrix = numpy.zeros((2,2))
    determinant = XX * n - X*X
    inverse_matrix[0,0] = XX / determinant
    inverse_matrix[0, 1] = -X / determinant
    inverse_matrix[1, 0] = -X / determinant
    inverse_matrix[1, 1] = n / determinant

    intercept = inverse_matrix[ 0,0 ] * Y + inverse_matrix[0,1] * XY
    slope = inverse_matrix[1, 0] * Y + inverse_matrix[1, 1] * XY

    return intercept, slope


def fill_zeros_or_nan_in_tail( variance_density:numpy.ndarray, frequencies:numpy.ndarray, power=-5, zero=0.0, points_in_fit=15 ):
    input_shape = variance_density.shape

    number_of_elements = numpy.prod(input_shape[:-1])
    number_of_frequencies = input_shape[-1]
    variance_density = variance_density.reshape( (number_of_elements,input_shape[-1]) )


    for ii in range( 0, number_of_elements ):
        is_zero = False
        index = number_of_frequencies - 1
        max = numpy.max(variance_density[ii,:])
        zero = max / 2**11

        for jj in range( number_of_frequencies-1,-1,-1 ):
            if variance_density[ii,jj] > zero:
                # Note, for nan values this is also always false. No need to seperately check for that.
                index = jj
                break
        else:
            # no valid value found, we cannot extrapolate. Technically this is not needed as we catch this below as
            # well (since index=number_of_frequencies by default). But to make more explicit that we catch for this
            # scenario I leave it in.
            continue

        if index == number_of_frequencies-1:
            continue
        elif index < 2*points_in_fit:
            continue

        f = frequencies[index-points_in_fit+1:index+1]
        e = variance_density[ii, index - points_in_fit + 1:index + 1]

        # Least squares fit of the slope
        intercept, power = least_squares(numpy.log(f),numpy.log(e))

        #e0 = numpy.exp(intercept)

        e0 = variance_density[ii, index]
        f0 = frequencies[index]
        e0 = numpy.exp(intercept)

        print(power,e0)
        for jj in range(index+1,number_of_frequencies):
            variance_density[ii,jj] = e0*(frequencies[jj]) **power

    return numpy.reshape(variance_density,input_shape)
