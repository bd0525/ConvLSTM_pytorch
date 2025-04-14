import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

from convlstm import ConvLSTM
from convlstm import ConvLSTMCell

# Create a video prediction model
class VideoFramePredictionModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=128, kernel_size=(3, 3), num_layers=2):
        super(VideoFramePredictionModel, self).__init__()
        
        # Encoder with BatchNorm - Simplified to preserve more spatial information
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # ConvLSTM
        self.convlstm = ConvLSTM(
            input_dim=16,
            hidden_dim=[hidden_dim] * num_layers,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # no upsampling needed
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # using tanh for sharper transitions
        )
        
    def forward(self, x, future_frames=1):
        batch_size, seq_len, c, h, w = x.size()
        
        # Encode each input frame
        encoded_frames = [self.encoder(x[:, t]) for t in range(seq_len)]
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        # Run through ConvLSTM
        layer_outputs, last_states = self.convlstm(encoded_sequence)
        
        outputs = []
        current_states = last_states
        
        # Use the encoded last observed frame as initial input
        last_encoded_frame = encoded_frames[-1]
        cur_input = last_encoded_frame
        
        # Generate future frames
        for _ in range(future_frames):
            next_states = []
            
            # Process through each ConvLSTM layer
            for layer_idx in range(self.convlstm.num_layers):
                h_cur, c_cur = current_states[layer_idx]
                h_next, c_next = self.convlstm.cell_list[layer_idx](cur_input, [h_cur, c_cur])
                next_states.append((h_next, c_next))
                cur_input = h_next
            
            current_states = next_states
            output = self.decoder(h_next)
            
            # Scale tanh output from [-1, 1] to [0, 1] for proper image visualization
            output = (output + 1) / 2
            
            # Re-encode the output for the next prediction
            cur_input = self.encoder(output)
            
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

def sharpness_loss(pred, target):
    """Loss function that promotes sharper edges in the predictions"""
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    
    # Reshape batch and sequence dimensions to apply filters
    pred_reshaped = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
    target_reshaped = target.view(-1, 1, target.shape[-2], target.shape[-1])
    
    # Apply filters
    pred_grad_x = F.conv2d(pred_reshaped, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_reshaped, sobel_y, padding=1)
    target_grad_x = F.conv2d(target_reshaped, sobel_x, padding=1)
    target_grad_y = F.conv2d(target_reshaped, sobel_y, padding=1)
    
    # Compute gradient magnitudes
    pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
    target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
    
    # Return MSE of gradient magnitudes
    return F.mse_loss(pred_grad_mag, target_grad_mag)

def combined_loss(pred, target, alpha=0.8):
    """Combines pixel-wise MSE with edge-aware sharpness loss"""
    mse = F.mse_loss(pred, target)
    sharp = sharpness_loss(pred, target)
    return alpha * mse + (1 - alpha) * sharp

# Create a custom dataset for video sequences
class VideoFrameDataset(Dataset):
    def __init__(self, data_path, seq_length=10, pred_length=1, transform=None):
        """
        Args:
            data_path: Directory containing video frames or sequences
            seq_length: Number of input frames to use
            pred_length: Number of future frames to predict
            transform: Optional transformations to apply
        """
        self.data_path = data_path
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.transform = transform
        
        # For simplicity, let's assume we have folders named 'sequence_1', 'sequence_2', etc.
        # Each containing frames as 'frame_001.png', 'frame_002.png', etc.
        self.sequences = [os.path.join(data_path, d) for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))]
        
        # Filter out sequences that are too short
        self.valid_sequences = []
        for seq in self.sequences:
            frames = sorted([f for f in os.listdir(seq) if f.endswith('.png') or f.endswith('.jpg')])
            if len(frames) >= seq_length + pred_length:
                self.valid_sequences.append((seq, frames))
                
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        seq_dir, frames = self.valid_sequences[idx]
        
        # Load input sequence
        input_seq = []
        for i in range(self.seq_length):
            img_path = os.path.join(seq_dir, frames[i])
            img = Image.open(img_path).convert('L')  # convert to grayscale
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                img = img.unsqueeze(0)  # add channel dimension
            input_seq.append(img)
            
        # Load target sequence (future frames)
        target_seq = []
        for i in range(self.seq_length, self.seq_length + self.pred_length):
            img_path = os.path.join(seq_dir, frames[i])
            img = Image.open(img_path).convert('L')
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                img = img.unsqueeze(0)
            target_seq.append(img)
            
        # Stack along time dimension
        input_seq = torch.stack(input_seq, dim=0)
        target_seq = torch.stack(target_seq, dim=0)
        
        return input_seq, target_seq

def train_model(model, train_loader, val_loader, num_epochs=15, patience=5, device='cuda'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in progress_bar:
            # Move inputs and targets to the correct shape and device
            inputs = inputs.to(device)  # [batch, seq_len, c, h, w]
            targets = targets.to(device)  # [batch, pred_len, c, h, w]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, future_frames=targets.size(1))
            
            # Calculate loss with the combined loss function
            loss = combined_loss(outputs, targets, alpha=0.8)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # increased threshold
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({"train_loss": loss.item()})
            
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs, future_frames=targets.size(1))
                loss = combined_loss(outputs, targets, alpha=0.8)
                
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(script_dir, 'best_video_prediction_model.pth'))
            patience_counter = 0
            print("Saved model checkpoint (improved validation loss)")
        else:
            patience_counter += 1
            print(f"Validation loss didn't improve. Patience: {patience_counter}/{patience}")
            
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(script_dir, 'training_loss_curve.png'))
    plt.close()
            
    return model

# Function to visualize predictions with numerical metrics
def visualize_predictions(model, test_loader, device='cuda', num_samples=5):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model.eval()
    samples_seen = 0
    
    # Initialize metrics
    total_mse = 0
    total_psnr = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions
            outputs = model(inputs, future_frames=targets.size(1))
            
            # Calculate metrics
            batch_mse = F.mse_loss(outputs, targets).item()
            batch_psnr = 10 * torch.log10(1 / batch_mse).item()
            
            total_mse += batch_mse
            total_psnr += batch_psnr
            
            # Visualize the results for each sample in the batch
            for b in range(min(inputs.size(0), num_samples - samples_seen)):
                # Create a figure with 3 rows: input, target, prediction
                num_cols = targets.size(1)
                fig, axes = plt.subplots(3, num_cols, figsize=(num_cols * 3, 9))
                
                # Handle the case when there's only one prediction frame
                if num_cols == 1:
                    axes = axes.reshape(3, 1)
                
                # First row: Show the last few input frames
                for t in range(min(num_cols, 3)):
                    idx = -min(num_cols, 3) + t
                    axes[0, t].imshow(inputs[b, idx, 0].cpu().numpy(), cmap='gray')
                    axes[0, t].set_title(f'Input t{idx}')
                    axes[0, t].axis('off')
                
                # Second row: Show target frames
                for t in range(num_cols):
                    target_frame = targets[b, t, 0].cpu().numpy()
                    axes[1, t].imshow(target_frame, cmap='gray')
                    axes[1, t].set_title(f'Target t+{t+1}')
                    axes[1, t].axis('off')
                
                # Third row: Show predicted frames
                for t in range(num_cols):
                    pred_frame = outputs[b, t, 0].cpu().numpy()
                    
                    # Calculate frame-specific metrics
                    frame_mse = F.mse_loss(outputs[b, t], targets[b, t]).item()
                    frame_psnr = 10 * torch.log10(torch.tensor(1.0) / torch.tensor(frame_mse)).item()
                    
                    axes[2, t].imshow(pred_frame, cmap='gray')
                    axes[2, t].set_title(f'Pred t+{t+1}\nMSE: {frame_mse:.4f}\nPSNR: {frame_psnr:.2f}')
                    axes[2, t].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(script_dir, f'prediction_sample_{samples_seen + b}.png'))
                plt.close()
                
            samples_seen += inputs.size(0)
            if samples_seen >= num_samples:
                break
    
    # Print average metrics
    avg_mse = total_mse / samples_seen
    avg_psnr = total_psnr / samples_seen
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")

# Generate more varied synthetic data
def generate_improved_synthetic_data(num_sequences=200, seq_length=20, height=64, width=64):
    """Generate more varied moving shapes for better training"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'synthetic_video_data')
    os.makedirs(data_dir, exist_ok=True)
    
    for seq_idx in range(num_sequences):
        seq_dir = os.path.join(data_dir, f'sequence_{seq_idx:03d}')
        os.makedirs(seq_dir, exist_ok=True)
        
        # Create a random moving shape with more variety
        shape_type = np.random.choice(['square', 'circle', 'triangle'])
        shape_size = np.random.randint(8, 25)  # More size variety
        
        # Initial position
        x, y = np.random.randint(0, width - shape_size), np.random.randint(0, height - shape_size)
        
        # Random velocity with more variation
        vx = np.random.uniform(-4, 4)
        vy = np.random.uniform(-4, 4)
        
        # Ensure at least some minimum velocity
        if abs(vx) < 1.0:
            vx = 1.0 if vx >= 0 else -1.0
        if abs(vy) < 1.0:
            vy = 1.0 if vy >= 0 else -1.0
        
        # Add slight acceleration for more complex motion
        ax = np.random.uniform(-0.1, 0.1)
        ay = np.random.uniform(-0.1, 0.1)
        
        for frame_idx in range(seq_length):
            # Create a blank frame
            frame = np.zeros((height, width), dtype=np.uint8)
            
            # Round position to integers
            x_int, y_int = int(round(x)), int(round(y))
            
            # Draw different shapes
            if shape_type == 'square':
                # Ensure within bounds
                x_int = max(0, min(x_int, width - shape_size))
                y_int = max(0, min(y_int, height - shape_size))
                frame[y_int:y_int+shape_size, x_int:x_int+shape_size] = 255
            
            elif shape_type == 'circle':
                # Create a meshgrid for the image
                yy, xx = np.mgrid[:height, :width]
                # Calculate distance from center
                circle = (xx - x_int) ** 2 + (yy - y_int) ** 2
                # Create the circle
                radius = shape_size // 2
                circle_mask = circle <= radius ** 2
                frame[circle_mask] = 255
            
            elif shape_type == 'triangle':
                # Define triangle vertices
                half_size = shape_size // 2
                vertices = np.array([
                    [x_int, y_int - half_size],  # top
                    [x_int - half_size, y_int + half_size],  # bottom left
                    [x_int + half_size, y_int + half_size],  # bottom right
                ])
                
                # Draw a filled triangle (simplified approach)
                # This fills a bounding rectangle and then masks out non-triangle parts
                min_x, min_y = np.min(vertices, axis=0)
                max_x, max_y = np.max(vertices, axis=0)
                
                # Ensure within image bounds
                min_x, min_y = max(0, min_x), max(0, min_y)
                max_x, max_y = min(width-1, max_x), min(height-1, max_y)
                
                # Fill the bounding rectangle
                frame[int(min_y):int(max_y), int(min_x):int(max_x)] = 255
            
            # Save the frame
            frame_path = os.path.join(seq_dir, f'frame_{frame_idx:03d}.png')
            Image.fromarray(frame).save(frame_path)
            
            # Update position with velocity
            x += vx
            y += vy
            
            # Update velocity with acceleration
            vx += ax
            vy += ay
            
            # Bounce if hitting the boundary
            if x <= 0 or x >= width - shape_size:
                vx = -vx
                x = max(0, min(x, width - shape_size))
            if y <= 0 or y >= height - shape_size:
                vy = -vy
                y = max(0, min(y, height - shape_size))
    
    return data_dir

# Main function
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Starting the improved video prediction application...")
    
    print("Generating synthetic data...")
    data_dir = generate_improved_synthetic_data(num_sequences=200, seq_length=20)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and data loaders
    print("Creating datasets...")
    # Input sequence of 10 frames, predict next 5 frames
    dataset = VideoFrameDataset(data_dir, seq_length=10, pred_length=5)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = VideoFramePredictionModel(input_channels=1, hidden_dim=128, num_layers=2)
    
    # Train the model with improved settings
    print("Training model...")
    model = train_model(model, train_loader, val_loader, num_epochs=15, patience=5, device=device)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(script_dir, 'best_video_prediction_model.pth')))
    
    # Visualize predictions with metrics
    print("Visualizing predictions...")
    visualize_predictions(model, test_loader, device=device)
    
    print("Done!")


if __name__ == '__main__':
    main()