

cdef enum:
    GSL_SUCCESS = 0
    GSL_EDOM = 1
    GSL_FAILURE = -1


cdef extern from "gsl/gsl_interp.h":
    ctypedef struct gsl_interp_accel
    ctypedef struct gsl_interp_type
    ctypedef struct gsl_interp
    gsl_interp_type * gsl_interp_linear

    int gsl_interp_init(gsl_interp * obj,  double xa[],  double ya[], size_t size) nogil

    gsl_interp_accel * gsl_interp_accel_alloc() nogil
    gsl_interp * gsl_interp_alloc( gsl_interp_type * T, size_t n) nogil
    void gsl_interp_free (gsl_interp * interp) nogil
    void gsl_interp_accel_free (gsl_interp_accel* acc)

    int gsl_interp_eval_e( gsl_interp * obj,
                double xa[],  double ya[], double x,
                gsl_interp_accel * a, double *y) nogil


cdef extern from "gsl/gsl_roots.h":

    ctypedef struct gsl_root_fsolver_type:
        char *name
        size_t size
        int (*set) (void *state, gsl_function * f, double * root, double x_lower, double x_upper) nogil
        int (*iterate) (void *state, gsl_function * f, double * root, double * x_lower, double * x_upper) nogil

    ctypedef struct gsl_root_fsolver:
        gsl_root_fsolver_type * type
        gsl_function * function
        double root
        double x_lower
        double x_upper
        void *state

    ctypedef struct gsl_function:
        double (* function) (double x, void * params) nogil
        void * params

    gsl_root_fsolver * gsl_root_fsolver_alloc ( gsl_root_fsolver_type * T) nogil
    void gsl_root_fsolver_free (gsl_root_fsolver * s) nogil

    int gsl_root_fsolver_set (gsl_root_fsolver * s, gsl_function * f,
                                double x_lower, double x_upper) nogil

    int gsl_root_fsolver_iterate (gsl_root_fsolver * s) nogil

    double gsl_root_fsolver_root ( gsl_root_fsolver * s) nogil
    double gsl_root_fsolver_x_lower ( gsl_root_fsolver * s) nogil
    double gsl_root_fsolver_x_upper ( gsl_root_fsolver * s) nogil

    int gsl_root_test_interval (double x_lower, double x_upper, double epsabs, double epsrel) nogil

    gsl_root_fsolver_type  * gsl_root_fsolver_bisection
    gsl_root_fsolver_type  * gsl_root_fsolver_brent
    gsl_root_fsolver_type  * gsl_root_fsolver_falsepos
