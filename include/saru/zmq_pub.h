/*** zmq_pub.h ***/
#pragma once
#include <zmq.h>
#include <cstdio>
#include <cstring>


extern void* g_zmq_ctx;
extern void* g_zmq_pub;


// 例: "tcp://*:5556" で bind（gNBが配信役）
//    "ipc:///tmp/srs_scs_pub.ipc" も高速で便利
inline void zmq_pub_init(const char* endpoint)
{
  g_zmq_ctx = zmq_ctx_new();
  if (!g_zmq_ctx) { std::perror("zmq_ctx_new"); std::abort(); }

  g_zmq_pub = zmq_socket(g_zmq_ctx, ZMQ_PUB);
  if (!g_zmq_pub) { std::perror("zmq_socket"); std::abort(); }

  int hwm = 2000;   zmq_setsockopt(g_zmq_pub, ZMQ_SNDHWM, &hwm, sizeof(hwm));   // バッファ上限
  int linger = 0;   zmq_setsockopt(g_zmq_pub, ZMQ_LINGER, &linger, sizeof(linger)); // 終了時に捨てる
  int timeo = 0;    zmq_setsockopt(g_zmq_pub, ZMQ_SNDTIMEO, &timeo, sizeof(timeo)); // 非ブロッキング

  if (zmq_bind(g_zmq_pub, endpoint) != 0) { std::perror("zmq_bind"); std::abort(); }
}

inline void zmq_pub_close()
{
  if (g_zmq_pub) { zmq_close(g_zmq_pub); g_zmq_pub = nullptr; }
  if (g_zmq_ctx) { zmq_ctx_term(g_zmq_ctx); g_zmq_ctx = nullptr; }
}
