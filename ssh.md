# SSH access to the Intel server — instructions for agents

This document tells an agent how to get SSH access to the team Intel server.

> **To the human:** get the private key file from a team member first. The key is **not** in this repo. Then tell your agent: "Follow ssh.md to get SSH access to the Intel server."

## Target

| Item | Value |
|---|---|
| SSH alias | `intel` (connect with `ssh intel`) |
| Server IP address | `10.0.40.65` |
| Port | `22` |
| User | `intel` |
| Authentication | SSH key only. Password login is off. |
| Network | Wi-Fi **Junction2026** only. The server is not on the internet. |

## Rules

- Do not ask for the private key text in chat. Ask only for the path to the key file.
- Do not show the private key in your output, and do not commit it to this repo.
- Do not turn off host key checks (`StrictHostKeyChecking no`).
- Do not change the setup on the server. Only make changes that the user asks for (for example, to install packages or `udev` rules). For these changes, you can use `sudo` (see [Sudo](#sudo)).
- Do not change the sudo, SSH, or network setup on the server.
- Use `-o BatchMode=yes` for SSH commands, so that SSH does not stop to wait for a password.

## Step 1 — Log in

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 intel 'whoami; hostname'
```

If the output is `intel` and `intel-XPS-16-DA16260`, access is ready. Tell the user: "Connect with `ssh intel`."

## Step 2 — If the login fails

**The network is slightly unreliable.** If the error is a network error (`timed out`, `No route to host`, `Network is unreachable`, `Connection reset`, `Connection closed`), wait 5 seconds and retry the Step 1 command. Retry **at most 3 times**. If all retries fail, ask the user to make sure that they are on the **Junction2026** Wi-Fi and that a VPN is not on, then stop.

For other errors:

| Error | Action |
|---|---|
| `Could not resolve hostname intel` | The alias is not set up. Do Step 3. |
| `Permission denied (publickey)`, or a problem with the key file (`no such identity`, `UNPROTECTED PRIVATE KEY FILE`, `invalid format`) | Do Step 3. |
| `Host key verification failed`, `REMOTE HOST IDENTIFICATION HAS CHANGED` | **Stop. Do not bypass this check.** Tell the user. |
| `Connection refused` | SSH does not run on the server. Stop and tell the user. |

## Step 3 — Setup

Do the setup, then go back to Step 1. You can do the setup again at any time.

1. **Find the private key.** If `~/.ssh/intel_junction_ed25519` exists from an earlier setup, use it. If not, ask the user for the path to the key file that they got from a team member.
2. **Install the key.** Copy the key to `~/.ssh/intel_junction_ed25519`. Make sure the file has Unix (LF) line endings. Set permissions so that only the user can read it: `chmod 600` on macOS and Linux. On Windows, remove inherited permissions and give access to the current user only.
3. **Pin the server host keys.** Write these lines to `~/.ssh/known_hosts_intel`:

   ```
   10.0.40.65 ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMANmLxaR5h/NpBVCBleO9GPDpjFvNIA1WmvVHTQEbttHf7E/XiL6QnY5nse/+4jKLSuL3zxAVw+cLSY/OIiVNo=
   10.0.40.65 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ6TrXHbtmjkCHVzFuhIdfIlNgha4beqaHIvM61EqxAP
   10.0.40.65 ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDDS2iw4gQ5lENNATZoKNxVzHOiuNe9VB8bpKCnnVmIAPslfU1tA66gwfxoRiHoOn1CFN8VELtWxu/VjaLJe2rikiChf7YovJupJR/6A1Alza2693eQaAhpyT+mx5HvzJDr0kDqyjKcS/lyKNXLTwMXqcMzBDog6CtIH10lpf+0/x5RvpdocnvmBq5ODyL6bAEXasPKPuLWWYDqqWST3DES83j82MLT3+pTeVrj88rvKYfUy2a/yS4WPGR351gmtdQcEBJTuViMvMULIJRa310e+357T5cObktLAEWyC9d4z+mZnYk7tc7cqpbp5Q53Mc8fP7Vn4GsIQtak+ckvKn3hX0WOSbCQCx/NOEEb76xRTO21VDGrXVdoWh07zfpOjWcNPpAMYBWSEpbQcEOqLkYUqPDC35QagwowXsBNfOXzwpGNgKx1VHGSUKL7oToM7MuESYfm95BmrBbCTuQr0c8uHxcwMIZQqh2yv5M2BOSgYStq9sXuzsDxZG/3hJQpEsk=
   ```

4. **Add the SSH alias.** Add this block at the **top** of `~/.ssh/config`. If an old `Host intel` block exists, remove it first. On Windows, use forward slashes in the paths.

   ```
   Host intel
       HostName 10.0.40.65
       User intel
       IdentityFile ~/.ssh/intel_junction_ed25519
       IdentitiesOnly yes
       UserKnownHostsFile ~/.ssh/known_hosts_intel
       StrictHostKeyChecking yes
       ServerAliveInterval 30
   ```

## Use the server

Run a command on the server:

```bash
ssh -o BatchMode=yes intel '<command>'
```

The network is not reliable, so start long-running commands in `tmux` or with `nohup`. Then a dropped connection does not stop them.

## Sudo

The `intel` user can use `sudo` without a password.

| Item | Value |
|---|---|
| Sudo rule file | `/etc/sudoers.d/90-intel-nopasswd` (owner `root:root`, mode `0440`) |
| Rule | `intel ALL=(ALL) NOPASSWD: ALL` |
| Groups | `intel` is also in the `dialout` group. This gives access to USB serial ports, for example the SO-101 robot arms on `/dev/ttyACM*`. |
| To remove the rule (only if the user asks) | `sudo rm /etc/sudoers.d/90-intel-nopasswd` |

- Use `sudo -n`. If `sudo` asks for a password, the command then fails and does not stop to wait:

  ```bash
  ssh -o BatchMode=yes intel 'sudo -n <command>'
  ```

- Use `sudo` only for tasks that the user asks for. Do not change the sudo, SSH, or network setup.

> **Important:** a person with the shared SSH key has full root access to the laptop.
