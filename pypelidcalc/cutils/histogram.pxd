cdef enum:
  SUCCESS = 0
  FAILURE  = -1

cdef int histogram_bin(double x, double start, double step) nogil

cpdef double [:] histogram(double [:] x, double [:] bins)

cdef int _histogram2d(double [:,:] x, double [:] bins_y, double [:] bins_x, double [:,:] h) nogil
cpdef double [:,:] histogram2d(double [:,:] x, double [:] bins_y, double [:] bins_x)
