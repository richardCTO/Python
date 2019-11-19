#%%
import numpy

# x, y data values
# Change these arrays for each answer
x = [1, 2, 3, 4, 5, 6]
y = [4, 3, 2, 9, 15, 28]

# Function to find coeficent b, coeficent a
# and Coeficent of determinatoin R-squared
def polyfit(x, y, degree):
    
    results = {}

    coeffs = numpy.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

print(polyfit( x, y, 1))

#Find Correlation Coeficent, r
correlation_coeficent = numpy.corrcoef(x, y)

# Print Correlation Coefficient r
print('Correlation Coefficient ', correlation_coeficent)

# %%
