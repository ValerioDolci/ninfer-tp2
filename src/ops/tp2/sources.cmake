# Two-device (tp 2) Op sources of ninfer_ops, included from ../basic_sources.cmake with one line.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../common/allreduce.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../common/peer_mailbox.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../launcher/argmax_split.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../launcher/concat_rows.cu"
)
