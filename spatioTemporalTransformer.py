# SpatioTemporalTransformer.py - Full Model with DCT, Progressive Decoding, Joint Embedding, and Memory Dictionary

import os
import math
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fftpack import dct

# === Config ===
T_OBS = 30
T_PRED = 40
T_TOTAL = T_OBS + T_PRED
NUM_JOINTS = 24
MEMORY_CELLS = 8
PROGRESSIVE_STAGES = [
    [0], [1], [2, 3, 4, 5], [6, 7, 8, 9],
    [10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]
]

# === DCT & iDCT ===
def apply_dct_2d(time_np):
    T, D = time_np.shape
    out = [dct(time_np[:, d], type=2, norm='ortho') for d in range(D)]
    return np.stack(out, axis=1)

def build_idct_matrix(N):
    mat = torch.zeros(N, N, dtype=torch.float32)
    for n in range(N):
        for k in range(N):
            alpha = math.sqrt(1.0/N) if k == 0 else math.sqrt(2.0/N)
            mat[n, k] = alpha * math.cos(math.pi*(2*n+1)*k/(2.0*N))
    return mat

def torch_idct_2d(freq, idct_mat):
    B, T, D = freq.shape
    freq = freq.permute(0, 2, 1).contiguous()
    out = torch.matmul(idct_mat, freq)
    return out.permute(0, 2, 1).contiguous()

# === Dataset ===
class MovementDataset(Dataset):
    def __init__(self, h5_path, T_obs=T_OBS, T_pred=T_PRED, subset_size=None):
        super().__init__()
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_total = T_obs + T_pred

        self.h5 = h5py.File(h5_path, 'r')
        all_keys = list(self.h5['movements'].keys())
        print(f"🔎 Total segments in HDF5: {len(all_keys)}")

        self.valid_samples = all_keys if not subset_size else all_keys[:subset_size]
        print(f"✅ Using {len(self.valid_samples)} segments")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        key = self.valid_samples[idx]
        grp = self.h5['movements'][key]
        angles = grp['angles'][:].astype(np.float32)
        valid_length = grp.attrs['valid_length']

        # Only take the valid part of the sequence
        angles = angles[:valid_length]

        if valid_length < self.T_total:
            raise ValueError(f"Sequence {key} is too short (has {valid_length} frames, need {self.T_total})")

        obs = angles[:self.T_obs]
        fut = angles[self.T_obs:self.T_total]

        # Optional: return metadata if needed
        metadata = {
            'movement_id': grp.attrs['movement_id'],
            'movement_name': grp.attrs['movement_name'],
            'session_id': grp.attrs['session_id'],
            'subject_id': grp.attrs['subject_id'],
            'exercise_id': grp.attrs['exercise_id'],
            'exercise_table': grp.attrs['exercise_table'],
            'repetition_id': grp.attrs['repetition_id'],
        }

        return obs, fut, metadata



# === Utility ===
def save_preprocessed_dataset(dataset, out_path="preprocessed_data.pt"):
    all_obs, all_fut = [], []
    for obs, fut in tqdm(dataset, desc="Saving preprocessed dataset"):
        all_obs.append(torch.tensor(obs))
        all_fut.append(torch.tensor(fut))
    torch.save((torch.stack(all_obs), torch.stack(all_fut)), out_path)
    print(f"💾 Saved preprocessed dataset to {out_path}")



# === Model ===
class HandDCTMotionModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, use_memory=True, progressive=True):
        super().__init__()
        self.use_memory = use_memory
        self.progressive = progressive
        self.d_model = d_model
        self.joint_embed = nn.Embedding(NUM_JOINTS, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, T_TOTAL * NUM_JOINTS, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder_proj = nn.Linear(d_model, 1)

        # Memory dictionary
        self.memory_keys = nn.Parameter(torch.randn(NUM_JOINTS, MEMORY_CELLS, d_model))
        self.memory_values = nn.Parameter(torch.randn(NUM_JOINTS, MEMORY_CELLS, d_model))

    def forward(self, x):
        B, T, J = x.shape
        joint_ids = torch.arange(J, device=x.device).unsqueeze(0).unsqueeze(0)
        joint_embs = self.joint_embed(joint_ids).expand(B, T, J, -1)
        x = x.unsqueeze(-1) + joint_embs
        x = x.view(B, T * J, -1)
        x = x + self.pos_encoder[:, :x.size(1), :]

        memory = self.encoder(x)
        output = torch.zeros_like(memory)
        if self.progressive:
            stages = PROGRESSIVE_STAGES
        else:
            stages = [list(range(NUM_JOINTS))]  # no staging

        for stage in stages:
            idxs = []
            for t in range(T):
                base = t * J
                for j in stage:
                    idxs.append(base + j)
            idx_t = torch.tensor(idxs, device=x.device)
            stage_input = memory[:, idx_t, :]

            # === Memory Attention ===
            if self.use_memory:
                joint_count = len(stage)
                time_steps = stage_input.shape[1] // joint_count
                bias_chunks = []
                for i, j in enumerate(stage):
                    keys = self.memory_keys[j]
                    values = self.memory_values[j]
                    context = stage_input[:, i*time_steps:(i+1)*time_steps, :]
                    attention = torch.softmax(torch.einsum('btd,cd->btc', context, keys), dim=-1)
                    retrieved = torch.einsum('btc,cd->btd', attention, values)
                    bias_chunks.append(retrieved)

                dict_bias = torch.cat(bias_chunks, dim=1)
            else:
                dict_bias = torch.zeros_like(stage_input)

            stage_input = stage_input + dict_bias  # safe out-of-place addition
            output[:, idx_t, :] = stage_input

        out = self.decoder_proj(output).squeeze(-1).view(B, T_TOTAL, J)
        out = out[:, -T_PRED:, :]  # Only keep the last T_PRED steps

        return out

# === Accuracy ===
THUMB_JOINTS = [2, 3, 4, 5, 6]

def compute_accuracy_deg(pred, target, threshold=10.0):
    mask = torch.ones(pred.shape[-1], dtype=torch.bool, device=pred.device)
    mask[THUMB_JOINTS] = False
    pred = pred[:, :, mask]
    target = target[:, :, mask]
    diff = (pred - target).abs()
    return (diff < threshold).float().mean().item()

# === Training ===
def train(epochs=10, h5_path="movement_level_dataset.h5", args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MovementDataset(h5_path, subset_size=args.subset_size)

    #subset = torch.utils.data.Subset(dataset, range(500))  # fast dev subset
    loader = DataLoader(
    torch.utils.data.Subset(dataset, range(min(len(dataset), args.subset_size)))
    if args.subset_size else dataset,
    batch_size=4,
    shuffle=True
)


    model = HandDCTMotionModel(
        use_memory=args.use_memory,
        progressive = not args.no_progressive
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    idct_mat = build_idct_matrix(T_PRED).to(device)
    losses, accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0

        for obs, fut in tqdm(loader, desc=f"Epoch {epoch+1} [Train]"):
            print("obs[0]:", obs[0].shape)
            print("fut[0]:", fut[0].shape)
            B = obs.size(0)
            rec_time_batch = []

            rec_time_batch = []

            for b in range(B):
                obs_b = obs[b].cpu().numpy()

                last_frame = obs_b[-1:]
                padded_obs = np.concatenate([obs_b, np.repeat(last_frame, T_PRED, axis=0)], axis=0)
                input_dct = apply_dct_2d(padded_obs)
                baseline_dct = input_dct[-T_PRED:]

                input_dct_t = torch.tensor(input_dct, dtype=torch.float32).unsqueeze(0).to(device)
                baseline_dct_t = torch.tensor(baseline_dct, dtype=torch.float32).unsqueeze(0).to(device)

                predicted_residual = model(input_dct_t)
                predicted_dct = predicted_residual + baseline_dct_t
                rec_time = torch_idct_2d(predicted_dct.permute(0, 2, 1), idct_mat).permute(0, 2, 1)

                rec_time_batch.append(rec_time)


            pred_fut = torch.cat(rec_time_batch, dim=0)  # (B, T_PRED, J)
            print("pred_fut[0]:", pred_fut[0, :, :3].cpu().detach().numpy())  # First 3 joints
            print("target_fut[0]:", fut[0, :, :3].cpu().numpy())


            # === Mask thumb joints
            mask = torch.ones(pred_fut.shape[-1], dtype=torch.bool, device=pred_fut.device)
            mask[THUMB_JOINTS] = False
            masked_pred = pred_fut[:, :, mask]
            masked_target = fut[:, :, mask].to(device)

            loss = loss_fn(masked_pred, masked_target)
            loss.backward()
            optimizer.step()

            acc = compute_accuracy_deg(pred_fut, fut.to(device))
            total_loss += loss.item()
            total_acc += acc


        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)
        losses.append(avg_loss)
        accs.append(avg_acc)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc*100:.2f}%")

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.subplot(2, 1, 2)
    plt.plot(accs)
    plt.title("Training Accuracy")
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("✅ Training complete. Saved plot as 'training_curves.png'")
    if args and args.save_model:
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 Model saved as 'best_model.pth'")


def test(h5_path="movement_level_dataset.h5", model_path="best_model.pth", args=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MovementDataset(h5_path, subset_size=args.subset_size)

    loader = DataLoader(
    torch.utils.data.Subset(dataset, range(min(len(dataset), args.subset_size)))
    if args.subset_size else dataset,
    batch_size=4,
    shuffle=True
    )


    model = HandDCTMotionModel(
        use_memory=args.use_memory,
        progressive=not args.no_progressive
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    idct_mat = build_idct_matrix(T_PRED).to(device)
    total_acc = 0
    num_samples = 0
    fig_idx = 0

    with torch.no_grad():
        for obs, fut in tqdm(loader, desc="Testing"):
            obs_np = obs[0].numpy()
            fut_np = fut[0].numpy()
            # === DEBUG: Inspect raw future joint values ===
            print("\n🧪 DEBUG: Checking ground truth sample")
            print("Ground truth (fut) shape:", fut[0].shape)
            print("Joint 9 min/max:", fut[0][:, 9].min().item(), fut[0][:, 9].max().item())
            print("Joint 9 first 5 values:", fut[0][:5, 9])
            # === DEBUG: Plot full sequence (obs + fut) for joint 9 ===

            full_seq = torch.cat([obs[0], fut[0]], dim=0).cpu().numpy()
            plt.figure(figsize=(8, 3))
            plt.plot(full_seq[:, 9])
            plt.axvline(x=30, color='gray', linestyle='--', label='OBS/PRED boundary')
            plt.title("Full Sequence (Joint 9) - Observed + Future")
            plt.legend()
            plt.savefig("debug_joint6_full_sequence.png")
            plt.close()
            print("📊 Saved debug plot: debug_joint6_full_sequence.png")


            # Pad input
            last_frame = obs_np[-1:]
            padded_obs = np.concatenate([obs_np, np.repeat(last_frame, T_PRED, axis=0)], axis=0)
            input_dct = apply_dct_2d(padded_obs)
            baseline_dct = input_dct[-T_PRED:]

            input_dct_t = torch.tensor(input_dct, dtype=torch.float32).unsqueeze(0).to(device)
            baseline_dct_t = torch.tensor(baseline_dct, dtype=torch.float32).unsqueeze(0).to(device)

            predicted_residual = model(input_dct_t)
            predicted_dct = predicted_residual + baseline_dct_t
            rec_time = torch_idct_2d(predicted_dct.permute(0, 2, 1), idct_mat).permute(0, 2, 1)
            pred_fut = rec_time

            acc = compute_accuracy_deg(pred_fut, fut[0].unsqueeze(0).to(device))
            total_acc += acc
            num_samples += 1

            # Save a comparison plot for the first few
            if fig_idx < 5:
                plot_joint_trajectories(pred_fut, fut[0].unsqueeze(0), joints_to_plot=[6, 7, 8], prefix="trajectories", sample_id=fig_idx)
                plt.figure(figsize=(12, 3))
                joint_idx = 6  # pick a central finger joint, change if needed
                plt.plot(fut_np[:, joint_idx], label="Ground Truth")
                plt.plot(pred_fut.cpu().numpy()[0, :, joint_idx], label="Prediction")
                plt.title(f"Prediction vs Ground Truth (Joint {joint_idx})")
                plt.legend()
                plt.savefig(f"prediction_vs_groundtruth_{fig_idx}.png")
                plt.close()
                fig_idx += 1

    print(f"✅ Test Accuracy (masked): {100 * total_acc / num_samples:.2f}%")



def plot_joint_trajectories(pred_fut, target_fut, joints_to_plot=None, prefix="trajectory", sample_id=0):
    """
    Plots predicted vs. ground truth joint angles over time.

    Args:
        pred_fut: Tensor of shape (1, T_PRED, J)
        target_fut: Tensor of shape (1, T_PRED, J)
        joints_to_plot: List of joint indices to plot (default: first 5)
    """


    pred_np = pred_fut.squeeze(0).cpu().numpy()
    target_np = target_fut.squeeze(0).cpu().numpy()
    T, J = pred_np.shape

    if joints_to_plot is None:
        joints_to_plot = list(range(min(5, J)))  # plot first 5 joints by default

    plt.figure(figsize=(12, 8))
    for i, j in enumerate(joints_to_plot):
        plt.subplot(len(joints_to_plot), 1, i + 1)
        plt.plot(target_np[:, j], label='Ground Truth')
        plt.plot(pred_np[:, j], label='Prediction')
        plt.title(f"Joint {j}")
        plt.ylabel("Angle (deg)")
        plt.legend()
    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(f"{prefix}_sample{sample_id}.png")
    print(f"🖼️ Saved joint trajectory plot: {prefix}_sample{sample_id}.png")
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data', type=str, default="full_shadow_dataset.h5")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--use-memory', action="store_true")
    parser.add_argument('--no-progressive', action="store_true")
    parser.add_argument('--save-model', action='store_true', help="Save the model after training")
    parser.add_argument('--save-dataset', action='store_true', help="Save filtered HDF5 movement keys")
    parser.add_argument('--load-dataset', action='store_true', help="Load filtered HDF5 movement keys")
    parser.add_argument('--save-preprocessed', action='store_true', help="Save full obs/fut tensors")
    parser.add_argument('--load-preprocessed', action='store_true', help="Load full obs/fut tensors")
    parser.add_argument('--subset-size', type=int, default=None, help="Use a subset of the dataset for faster debugging")

    args = parser.parse_args()

    if args.mode == "train":
        if args.load_preprocessed:
            dataset = MovementDataset(args.data, load_preprocessed=True, subset_size=args.subset_size)
        else:
            dataset = MovementDataset(
                args.data,
                use_cache=args.save_dataset or args.load_dataset,
                subset_size=args.subset_size
            )
            if args.save_preprocessed:
                save_preprocessed_dataset(dataset)

        model = HandDCTMotionModel(
            use_memory=args.use_memory,
            progressive=not args.no_progressive
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        train(epochs=args.epochs, h5_path=args.data, args=args)

        if args.save_model:
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Model saved as 'best_model.pth'")

    elif args.mode == "test":
        test(h5_path=args.data, args=args)



