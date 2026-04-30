# Bringing up Lancaster VPN on this Linux box

This Linux workstation (`LU-C09RH34`, on private subnet `10.61.8.0/21`) cannot
reach `wayland-2022.hec.lancaster.ac.uk:22` directly — the campus firewall
blocks outbound SSH from this VLAN. To `rsync` checkpoints to the HEC cluster
you must first bring up the Lancaster VPN tunnel.

## One-time setup

### 1. Install `openconnect`

`openvpn` is already installed but Lancaster ISS uses an AnyConnect-compatible
TLS VPN; `openconnect` is the right client.

```bash
sudo apt update
sudo apt install -y openconnect
```

### 2. Test reachability of the VPN endpoint

```bash
timeout 5 bash -c "echo > /dev/tcp/vpn.lancs.ac.uk/443" && echo OPEN
```
This should print `OPEN`. (DNS already resolves `vpn.lancs.ac.uk` to
`148.88.248.108`.)

## Each-time connection

```bash
sudo openconnect --user=xiar3 vpn.lancs.ac.uk
```
You'll be prompted for your Lancaster password and (likely) MFA. Leave the
process running in a terminal — closing it tears down the tunnel.

If Lancaster ISS uses GlobalProtect rather than AnyConnect, force the
protocol:
```bash
sudo openconnect --protocol=gp --user=xiar3 vpn.lancs.ac.uk
```

### Verify the tunnel

In a *second* terminal (the first is occupied by openconnect):

```bash
# 1. Check a tun interface exists:
ip -4 addr | grep -E "tun|utun"

# 2. Check the cluster is now reachable:
timeout 5 bash -c "echo > /dev/tcp/wayland-2022.hec.lancaster.ac.uk/22" && echo OPEN

# 3. Try SSH:
ssh -o ConnectTimeout=8 xiar3@wayland-2022.hec.lancaster.ac.uk hostname
```

If step 2 prints `OPEN` and step 3 returns the cluster's hostname, you're
ready to rsync.

## Run the rsync

```bash
cd /home/xiar3/experiments/STVqvae
bash paper_assets/scripts/sync_checkpoints.sh --check    # preflight
bash paper_assets/scripts/sync_checkpoints.sh --dry-run  # see what will move
bash paper_assets/scripts/sync_checkpoints.sh            # do it (~15.5 GB)
```

Expected runtime depends on bandwidth: at 100 Mbit/s ≈ 25 min, at 1 Gbit/s
≈ 2.5 min. The script uses `--partial --inplace` so a dropped connection
can be resumed by re-running it.

## Skipping passwords each rsync chunk

If you don't already have an SSH key authorised on the cluster:

```bash
ssh-keygen -t ed25519       # if you don't have one yet
ssh-copy-id xiar3@wayland-2022.hec.lancaster.ac.uk
```

Now `rsync` won't prompt for a password during the transfer.

## Troubleshooting

- **`openconnect: Connection refused`** — the VPN endpoint may be a different
  hostname for staff vs students; check Lancaster ISS docs at
  https://answers.lancaster.ac.uk/display/ISS/VPN.
- **`tun device not found`** — load the kernel module: `sudo modprobe tun`.
- **`rsync: connection unexpectedly closed`** — VPN dropped mid-transfer.
  `--partial --inplace` makes re-running the script resume from where it
  stopped; just rerun.
- **MFA loops** — some Lancaster VPN setups require Microsoft Authenticator;
  approve the push within the time limit or `openconnect` exits.

## After the transfer

The cluster will have:

```
/mmfs1/storage/users/xiar3/exp/STVqvae/discrete_mbrl/model_free/models/
├── MiniGrid-DoorKey-8x8-v0/
│   ├── sweep_dk8_v2_s4_best_model.pt        ── sweep_dk8_v2_s5_best_model.pt
│   ├── sweep_dk8_v5dc_s4_best_model.pt      ── sweep_dk8_v5dc_s5_best_model.pt
│   ├── sweep_dk8_v6_s4_best_model.pt        ── sweep_dk8_v6_s5_best_model.pt
│   ├── sweep_dk8_vae_s1_best_model.pt       ── sweep_dk8_vae_s5_best_model.pt
│   └── sweep_doorkey8_{v2,v5,v6,v9}_s{1,2,3}_best_model.pt
└── MiniGrid-DoorKey-16x16-v0/
    ├── sweep_doorkey16_v6_s{1,2,3}_best_model.pt
    └── sweep_doorkey16_v5dc_s{1,2,3}_best_model.pt
```

(20 DK-8 checkpoints + 6 DK-16 = 29 files total — but only 26 contribute
to the pooled correlation; the 3 v5dc DK-16 collapsed runs are excluded
per the paper's §4.6.)
