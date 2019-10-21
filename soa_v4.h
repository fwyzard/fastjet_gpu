/* noted and future developments
 *
 *   - add const interfaces
 *   - add support for dynamic memory allocation
 *   - add support for pseudo-element assignments (e.g. soa[i] = soa[j])
 *   - add a concrete element type, with the same (column) fields as the soa, and copy/assignmant to/from the pseudo-element
 *   - add a concrete scalar type, with the same scalar fields as the soa, and copy/assignmant to/from the soa
 *
 */

#include <iostream>

#include <boost/preprocessor.hpp>

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_HOST_DEVICE __host__ __device__
#else
#define SOA_HOST_ONLY
#define SOA_HOST_DEVICE
#endif

// compile-time sized SoA

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one vale per element) */

#define SoA_scalar(TYPE, NAME) (0, TYPE, NAME)
#define SoA_column(TYPE, NAME) (1, TYPE, NAME)
 

/* declare SoA data members; these should exapnd to, for columns:
 *
 *   alignas(ALIGN) double x_[SIZE];
 *
 * and for scalars:
 *
 *   double x_;
 *
 */

#define _DECLARE_SOA_DATA_MEMBER_IMPL(IS_COLUMN, TYPE, NAME)                                                                        \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    alignas(ALIGN) TYPE BOOST_PP_CAT(NAME, _[SIZE]);                                                                                \
  ,                                                                                                                                 \
    TYPE BOOST_PP_CAT(NAME, _);                                                                                                     \
  )

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME)                                                                                \
  BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)

#define _DECLARE_SOA_DATA_MEMBERS(...)                                                                                              \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_DATA_MEMBER, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


/* declare SoA accessors; these should expand to, for columns:
 *
 *   double* x() { return x_; }
 *
 * and for scalars:
 *
 *   double& x() { return x_; }
 *
 */

#define _DECLARE_SOA_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                           \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE* NAME() { return BOOST_PP_CAT(NAME, _); }                                                                                  \
  ,                                                                                                                                 \
    TYPE& NAME() { return BOOST_PP_CAT(NAME, _); }                                                                                  \
  )

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME)                                                                                   \
  BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_ACCESSORS(...)                                                                                                 \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_ACCESSOR, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                     \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const* NAME() const { return BOOST_PP_CAT(NAME, _); }                                                                      \
  ,                                                                                                                                 \
    TYPE const& NAME() const { return BOOST_PP_CAT(NAME, _); }                                                                      \
  )

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME)                                                                             \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_CONST_ACCESSORS(...)                                                                                           \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_CONST_ACCESSOR, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


/* declare AoS-like element accessors; these should expand to, for columns:
 *
 *   double & x() { return * (soa_.x() + index_); }
 *
 * and for scalars:
 *
 *   double & x() { return soa_.x(); }
 *
 */

#define _DECLARE_SOA_ELEMENT_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                   \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE & NAME() { return * (soa_. NAME () + index_); }                                                                            \
  ,                                                                                                                                 \
    TYPE & NAME() { return soa_. NAME (); }                                                                                         \
  )

#define _DECLARE_SOA_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                           \
  BOOST_PP_EXPAND(_DECLARE_SOA_ELEMENT_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_ELEMENT_ACCESSORS(...)                                                                                         \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_ELEMENT_ACCESSOR, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                             \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const & NAME() const { return * (soa_. NAME () + index_); }                                                                \
  ,                                                                                                                                 \
    TYPE const & NAME() const { return soa_. NAME (); }                                                                             \
  )

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                     \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSORS(...)                                                                                   \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


/* assignment between elements; these should expand to, for columns:
 *
 *   x() = other.x()
 *
 * and, for scalars, an empty macro
 */

#define _DECLARE_SOA_ELEMENT_TO_ELEMENT_ASSIGNMENT_IMPL(IS_COLUMN, TYPE, NAME)                                                      \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    NAME () = other. NAME();                                                                                                        \
  ,                                                                                                                                 \
  )

#define _DECLARE_SOA_ELEMENT_TO_ELEMENT_ASSIGNMENT(R, DATA, TYPE_NAME)                                                              \
  BOOST_PP_EXPAND(_DECLARE_SOA_ELEMENT_TO_ELEMENT_ASSIGNMENT_IMPL TYPE_NAME)

#define _DECLARE_SOA_ELEMENT_TO_ELEMENT_ASSIGNMENTS(...)                                                                            \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_ELEMENT_TO_ELEMENT_ASSIGNMENT, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


/* assign a value_type to an element; these should expand to, for columns:
 *
 *   x() = value.x
 *
 * and, for scalars, an empty macro
 */

#define _DECLARE_SOA_VALUE_TO_ELEMENT_ASSIGNMENT_IMPL(IS_COLUMN, TYPE, NAME)                                                        \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    NAME () = value. NAME;                                                                                                          \
  ,                                                                                                                                 \
  )

#define _DECLARE_SOA_VALUE_TO_ELEMENT_ASSIGNMENT(R, DATA, TYPE_NAME)                                                                \
  BOOST_PP_EXPAND(_DECLARE_SOA_VALUE_TO_ELEMENT_ASSIGNMENT_IMPL TYPE_NAME)

#define _DECLARE_SOA_VALUE_TO_ELEMENT_ASSIGNMENTS(...)                                                                              \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_VALUE_TO_ELEMENT_ASSIGNMENT, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


/* evaluate an element into a value_type; these should expand to, for columns:
 *
 *   value.x = element.x()
 *
 * and, for scalars, an empty macro
 */

#define _DECLARE_SOA_ELEMENT_EVAL_IMPL(IS_COLUMN, TYPE, NAME)                                                                       \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    value. NAME = NAME ();                                                                                                          \
  ,                                                                                                                                 \
  )

#define _DECLARE_SOA_ELEMENT_EVAL(R, DATA, TYPE_NAME)                                                                               \
  BOOST_PP_EXPAND(_DECLARE_SOA_ELEMENT_EVAL_IMPL TYPE_NAME)

#define _DECLARE_SOA_ELEMENT_EVALS(...)                                                                                             \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_ELEMENT_EVAL, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


/* declare AoS-like value type
 */

#define _DECLARE_SOA_VALUE_TYPE_IMPL(IS_COLUMN, TYPE, NAME)                                                                         \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE NAME;                                                                                                                      \
  ,                                                                                                                                 \
  )

#define _DECLARE_SOA_VALUE_TYPE(R, DATA, TYPE_NAME)                                                                                 \
  BOOST_PP_EXPAND(_DECLARE_SOA_VALUE_TYPE_IMPL TYPE_NAME)

#define _DECLARE_SOA_VALUE_TYPES(...)                                                                                               \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_VALUE_TYPE, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))



/* dump SoA fields information; these should expand to, for columns:
 *
 *   std::cout << "  x_[" << SIZE << "] at " 
 *             << offsetof(self_type, x_) << " has size " << sizeof(x_) << std::endl;
 *
 * and for scalars:
 *
 *   std::cout << "  x_ at " 
 *             << offsetof(self_type, x_) << " has size " << sizeof(x_) << std::endl;
 *
 */

#define _DECLARE_SOA_DUMP_INFO_IMPL(IS_COLUMN, TYPE, NAME)                                                                          \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_[" << SIZE << "] at "                                                              \
              << offsetof(self_type, BOOST_PP_CAT(NAME, _)) << " has size " << sizeof(BOOST_PP_CAT(NAME, _)) << std::endl;          \
  ,                                                                                                                                 \
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_ at "                                                                              \
              << offsetof(self_type, BOOST_PP_CAT(NAME, _)) << " has size " << sizeof(BOOST_PP_CAT(NAME, _)) << std::endl;          \
  )

#define _DECLARE_SOA_DUMP_INFO(R, DATA, TYPE_NAME)                                                                                  \
  BOOST_PP_EXPAND(_DECLARE_SOA_DUMP_INFO_IMPL TYPE_NAME)

#define _DECLARE_SOA_DUMP_INFOS(...)                                                                                                \
  BOOST_PP_SEQ_FOR_EACH(_DECLARE_SOA_DUMP_INFO, CLASS, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))


#define declare_SoA_template(CLASS, ...)                                                                                            \
template <int SIZE, int ALIGN=0>                                                                                                    \
struct CLASS {                                                                                                                      \
                                                                                                                                    \
  /* these could be moved to an external type trait to free up the symbol names */                                                  \
  using self_type = CLASS;                                                                                                          \
  static const int size = SIZE;                                                                                                     \
  static const int alignment = ALIGN;                                                                                               \
                                                                                                                                    \
  /* introspection */                                                                                                               \
  SOA_HOST_ONLY                                                                                                                     \
  static void dump() {                                                                                                              \
    std::cout << #CLASS "<" << SIZE << ", " << ALIGN << "): " << std::endl;                                                         \
    std::cout << "  sizeof(...): " << sizeof(CLASS) << std::endl;                                                                   \
    std::cout << "  alignof(...): " << alignof(CLASS) << std::endl;                                                                 \
    _DECLARE_SOA_DUMP_INFOS(__VA_ARGS__)                                                                                            \
    std::cout << std::endl;                                                                                                         \
  }                                                                                                                                 \
                                                                                                                                    \
  /* struct to hold an individual element */                                                                                        \
  struct value_type {                                                                                                               \
    _DECLARE_SOA_VALUE_TYPES(__VA_ARGS__)                                                                                           \
  };                                                                                                                                \
                                                                                                                                    \
  /* AoS-like accessor to individual elements */                                                                                    \
  struct const_element {                                                                                                            \
    SOA_HOST_DEVICE                                                                                                                 \
    const_element(CLASS const& soa, int index) :                                                                                    \
      soa_(soa),                                                                                                                    \
      index_(index)                                                                                                                 \
    { }                                                                                                                             \
                                                                                                                                    \
    _DECLARE_SOA_CONST_ELEMENT_ACCESSORS(__VA_ARGS__)                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    value_type eval() const                                                                                                         \
    {                                                                                                                               \
      value_type value;                                                                                                             \
      _DECLARE_SOA_ELEMENT_EVALS(__VA_ARGS__)                                                                                       \
      return value;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
  private:                                                                                                                          \
    CLASS const& soa_;                                                                                                              \
    const int index_;                                                                                                               \
  };                                                                                                                                \
                                                                                                                                    \
  struct element {                                                                                                                  \
    SOA_HOST_DEVICE                                                                                                                 \
    element(CLASS & soa, int index) :                                                                                               \
      soa_(soa),                                                                                                                    \
      index_(index)                                                                                                                 \
    { }                                                                                                                             \
                                                                                                                                    \
    _DECLARE_SOA_ELEMENT_ACCESSORS(__VA_ARGS__)                                                                                     \
                                                                                                                                    \
    _DECLARE_SOA_CONST_ELEMENT_ACCESSORS(__VA_ARGS__)                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    element& operator=(element const& other)                                                                                        \
    {                                                                                                                               \
      _DECLARE_SOA_ELEMENT_TO_ELEMENT_ASSIGNMENTS(__VA_ARGS__)                                                                      \
      return *this;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    element& operator=(const_element const& other)                                                                                  \
    {                                                                                                                               \
      _DECLARE_SOA_ELEMENT_TO_ELEMENT_ASSIGNMENTS(__VA_ARGS__)                                                                      \
      return *this;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    element& operator=(value_type const& value)                                                                                     \
    {                                                                                                                               \
      _DECLARE_SOA_VALUE_TO_ELEMENT_ASSIGNMENTS(__VA_ARGS__)                                                                        \
      return *this;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    value_type eval() const                                                                                                         \
    {                                                                                                                               \
      value_type value;                                                                                                             \
      _DECLARE_SOA_ELEMENT_EVALS(__VA_ARGS__)                                                                                       \
      return value;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
  private:                                                                                                                          \
    CLASS & soa_;                                                                                                                   \
    const int index_;                                                                                                               \
  };                                                                                                                                \
                                                                                                                                    \
  /* AoS-like accessor */                                                                                                           \
  SOA_HOST_DEVICE                                                                                                                   \
  element operator[](int index) { return element(*this, index); }                                                                   \
                                                                                                                                    \
  SOA_HOST_DEVICE                                                                                                                   \
  const_element operator[](int index) const { return const_element(*this, index); }                                                 \
                                                                                                                                    \
  /* accessors */                                                                                                                   \
  _DECLARE_SOA_ACCESSORS(__VA_ARGS__)                                                                                               \
  _DECLARE_SOA_CONST_ACCESSORS(__VA_ARGS__)                                                                                         \
                                                                                                                                    \
private:                                                                                                                            \
  /* data members */                                                                                                                \
  _DECLARE_SOA_DATA_MEMBERS(__VA_ARGS__)                                                                                            \
                                                                                                                                    \
}
