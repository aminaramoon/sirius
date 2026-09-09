# Keep the parser aligned with Pixi without updating unrelated baseline ports.
vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  yhirose/cpp-httplib
  REF
  "v${VERSION}"
  SHA512
  809d55146f1ccdd00c78c8565691c31ad230dff51d9cdd1d34b45e65127f41dfe2b9bfe1cd5d27fd892da61d120f39ae8af18cae1a85329c2e8efcaf1b15f285
  HEAD_REF
  master)

set(VCPKG_BUILD_TYPE release) # Header-only, without optional client features.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}"
  OPTIONS
  -DHTTPLIB_COMPILE=OFF
  -DHTTPLIB_USE_OPENSSL_IF_AVAILABLE=OFF
  -DHTTPLIB_USE_ZLIB_IF_AVAILABLE=OFF
  -DHTTPLIB_USE_BROTLI_IF_AVAILABLE=OFF
  -DHTTPLIB_USE_ZSTD_IF_AVAILABLE=OFF)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME httplib CONFIG_PATH lib/cmake/httplib)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
