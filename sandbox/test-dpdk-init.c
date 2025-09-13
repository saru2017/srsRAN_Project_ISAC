#define _GNU_SOURCE
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>

static volatile int keep_running = 1;
static void handle_sigint(int sig) { (void)sig; keep_running = 0; }

static void print_fec(uint16_t port_id, const char *tag) {
    uint32_t fec = 0;
    int r = rte_eth_fec_get(port_id, &fec);
    if (r == 0) {
        printf("[%s] FEC mode now: ", tag);
        if (fec & RTE_ETH_FEC_AUTO)  printf("AUTO ");
        if (fec & RTE_ETH_FEC_RS)    printf("RS ");
        if (fec & RTE_ETH_FEC_BASER) printf("BASE-R ");
        if (fec & RTE_ETH_FEC_NOFEC) printf("NOFEC ");
        puts("");
    } else {
        printf("[%s] FEC get not supported (ret=%d)\n", tag, r);
    }
}

int main(int argc, char **argv) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_panic("EAL init failed\n");
    argc -= ret; argv += ret;

    const uint16_t port_id = 0;
    const uint16_t nb_rxq = 1, nb_txq = 1;
    const uint16_t desired_mtu = 9200;

    if (rte_eth_dev_count_avail() == 0)
        rte_panic("No available DPDK ports\n");

    // 念のため停止（testpmdの `port stop 0` 相当）
    if (rte_eth_dev_is_valid_port(port_id)) {
        rte_eth_dev_stop(port_id);
    }

    struct rte_eth_dev_info dev_info;
    rte_eth_dev_info_get(port_id, &dev_info);

    // 10Gbps/Full に固定（オートネゴ無効）
    struct rte_eth_conf port_conf = {0};
    port_conf.link_speeds = RTE_ETH_LINK_SPEED_10G | RTE_ETH_LINK_SPEED_FIXED;

    // JumboはMTUで指定
    port_conf.rxmode.mtu = desired_mtu;

    // mbufプール（16KB data roomで9Kを1セグ受信）
    const unsigned nb_mbuf = 8192;
    const uint32_t data_room = 16384;
    struct rte_mempool *mp = rte_pktmbuf_pool_create(
        "MBUF_POOL", nb_mbuf, 256, 0, data_room, rte_socket_id());
    if (mp == NULL) rte_panic("mbuf pool create failed\n");

    // デバイス設定
    ret = rte_eth_dev_configure(port_id, nb_rxq, nb_txq, &port_conf);
    if (ret < 0) rte_panic("dev configure failed: %d\n", ret);

    // RX/TXキュー
    uint16_t rx_desc = 1024, tx_desc = 1024;
    struct rte_eth_rxconf rxq_conf = dev_info.default_rxconf;
    struct rte_eth_txconf txq_conf = dev_info.default_txconf;

    ret = rte_eth_rx_queue_setup(port_id, 0, rx_desc,
                                 rte_eth_dev_socket_id(port_id), &rxq_conf, mp);
    if (ret < 0) rte_panic("rx queue setup failed: %d\n", ret);

    ret = rte_eth_tx_queue_setup(port_id, 0, tx_desc,
                                 rte_eth_dev_socket_id(port_id), &txq_conf);
    if (ret < 0) rte_panic("tx queue setup failed: %d\n", ret);

    // MTU=9200 を明示
    ret = rte_eth_dev_set_mtu(port_id, desired_mtu);
    if (ret < 0) {
        fprintf(stderr, "WARN: set MTU(%u) failed: %d\n", desired_mtu, ret);
    }

    // ★ FECをOFFに（NOFEC）。起動前に行うのが確実。
    //    未サポートなら負の戻り値（-ENOTSUP など）。
    int fec_ret = rte_eth_fec_set(port_id, RTE_ETH_FEC_NOFEC);
    if (fec_ret < 0) {
        fprintf(stderr, "WARN: FEC set NOFEC failed: %d (driver/firmware may not allow)\n",
                fec_ret);
    }
    print_fec(port_id, "after set");

    // 起動（testpmdの `port start 0`）
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) rte_panic("port start failed: %d\n", ret);

    struct rte_eth_link link;
    rte_eth_link_get_nowait(port_id, &link);
    printf("Port %u: link %s, speed %u Mbps, %s-duplex\n",
           port_id,
           link.link_status ? "UP" : "DOWN",
           link.link_speed,
           (link.link_duplex == RTE_ETH_LINK_FULL_DUPLEX) ? "full" : "half");

    signal(SIGINT, handle_sigint);
    printf("Running... (Ctrl-C to stop)\n");
    while (keep_running) sleep(1);

    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);
    rte_eal_cleanup();
    return 0;
}