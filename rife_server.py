import socket
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
import logging
from torch.nn import functional as F
import warnings
import time
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the RIFE repository
RIFE_REPO_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(RIFE_REPO_PATH, 'model'))

class RIFEInference:
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        # Use the train_log directory in the RIFE repository
        self.model_dir = os.path.join(RIFE_REPO_PATH, 'train_log')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Check for the main model file
        model_file = 'flownet.pkl'
        if not os.path.exists(os.path.join(self.model_dir, model_file)):
            logger.error(f"Missing model file: {model_file}")
            logger.error(f"Please download the model file from https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HpcMHqEMo")
            logger.error(f"and place it in: {self.model_dir}")
            raise FileNotFoundError(f"Missing model file in {self.model_dir}")
        
        # Try to load different model versions
        try:
            try:
                try:
                    from model.RIFE_HDv2 import Model
                    self.model = Model()
                    self.model.load_model(self.model_dir, -1)
                    print("Loaded v2.x HD model.")
                    logger.info("Loaded v2.x HD model.")
                except:
                    from train_log.RIFE_HDv3 import Model
                    self.model = Model()
                    self.model.load_model(self.model_dir, -1)
                    print("Loaded v3.x HD model.")
                    logger.info("Loaded v3.x HD model.")
            except:
                from model.RIFE_HD import Model
                self.model = Model()
                self.model.load_model(self.model_dir, -1)
                print("Loaded v1.x HD model")
                logger.info("Loaded v1.x HD model")
        except:
            from model.RIFE import Model
            self.model = Model()
            self.model.load_model(self.model_dir, -1)
            print("Loaded ArXiv-RIFE model")
            logger.info("Loaded ArXiv-RIFE model")
        
        self.model.eval()
        self.model.device()

    def preprocess_frame(self, frame):
        """Convert OpenCV frame to tensor format."""
        # Convert to float32 and normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Get dimensions
        h, w = frame.shape[:2]
        
        # Calculate padding to make dimensions multiples of 32
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        
        # Create padded frame
        padded_frame = np.zeros((ph, pw, 3), dtype=np.float32)
        padded_frame[:h, :w] = frame
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(padded_frame.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        return tensor, h, w, ph, pw

    def postprocess_frame(self, tensor, h, w):
        """Convert tensor back to OpenCV frame format."""
        # Convert to numpy and transpose back to HWC
        frame = tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # Crop to original size and convert to uint8
        frame = (frame[:h, :w] * 255).astype(np.uint8)
        
        return frame

    def interpolate(self, frame1, frame2, factor=0.5):
        """Interpolate between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            factor: Interpolation factor (0 to 1), default 0.5 for middle frame
        """
        # Preprocess frames
        frame1_tensor, h, w, ph, pw = self.preprocess_frame(frame1)
        frame2_tensor, _, _, _, _ = self.preprocess_frame(frame2)
        
        # Ensure both tensors have the same size
        if frame1_tensor.shape != frame2_tensor.shape:
            # Use the larger size for both tensors
            max_h = max(frame1_tensor.shape[2], frame2_tensor.shape[2])
            max_w = max(frame1_tensor.shape[3], frame2_tensor.shape[3])
            
            # Pad both tensors to the same size
            frame1_tensor = F.pad(frame1_tensor, (0, max_w - frame1_tensor.shape[3], 0, max_h - frame1_tensor.shape[2]))
            frame2_tensor = F.pad(frame2_tensor, (0, max_w - frame2_tensor.shape[3], 0, max_h - frame2_tensor.shape[2]))
        
        # Run inference
        with torch.no_grad():
            interpolated_tensor = self.model.inference(frame1_tensor, frame2_tensor, factor)
        
        # Postprocess result
        return self.postprocess_frame(interpolated_tensor, h, w)

def make_inference(rife, I0, I1, n):
    """Recursively generate intermediate frames using RIFE.
    
    Args:
        rife: RIFEInference instance
        I0: First frame
        I1: Last frame
        n: Number of intermediate frames to generate
    """
    middle = rife.interpolate(I0, I1, 0.5)
    if n == 1:
        return [middle]
    first_half = make_inference(rife, I0, middle, n=n//2)
    second_half = make_inference(rife, middle, I1, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def process_request(frame1_path, frame3_path, output_path, exp=1):
    """Process interpolation request using RIFE.
    
    Args:
        frame1_path: Path to first frame
        frame3_path: Path to last frame
        output_path: Path to save interpolated frame
        exp: Number of interpolation steps:
            0: Legacy mode (replace frame2 with interpolated frame)
            1: One interpolated frame (double FPS)
            >1: Multiple interpolated frames (2^exp - 1 frames)
    """
    try:
        # Read input frames
        frame1 = cv2.imread(frame1_path)
        frame3 = cv2.imread(frame3_path)
        
        if frame1 is None or frame3 is None:
            return "ERROR: Failed to read input frames"
        
        # Run interpolation
        rife = RIFEInference()
        
        if exp == 0:
            # Legacy mode: replace frame2 with interpolated frame
            middle = rife.interpolate(frame1, frame3, 0.5)
            # Save only the middle frame without index
            cv2.imwrite(output_path, middle)
        elif exp == 1:
            # One interpolated frame mode
            middle = rife.interpolate(frame1, frame3, 0.5)
            frames = [frame1, middle, frame3]
            # Save all frames with indices
            for i, frame in enumerate(frames):
                output_path_i = output_path.replace('.png', f'_{i}.png')
                cv2.imwrite(output_path_i, frame)
        else:
            # Generate frames using recursive approach
            frames = [frame1]
            intermediate_frames = make_inference(rife, frame1, frame3, 2**exp - 1)
            frames.extend(intermediate_frames)
            frames.append(frame3)
            
            # Save all frames with indices
            for i, frame in enumerate(frames):
                output_path_i = output_path.replace('.png', f'_{i}.png')
                cv2.imwrite(output_path_i, frame)
            
        return "OK"
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return f"ERROR: {str(e)}"

def check_port_in_use(port, host='localhost'):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True

def find_available_port(start_port=50051, max_attempts=10):
    """Find an available port starting from start_port."""
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        if not check_port_in_use(port):
            return port
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def kill_previous_instance(port):
    """Try to identify and kill a process running on the specified port (platform-specific)."""
    try:
        if sys.platform == 'win32':
            # For Windows
            os.system(f'for /f "tokens=5" %p in (\'netstat -ano ^| findstr :{port}\') do taskkill /F /PID %p')
        else:
            # For Linux/Mac
            os.system(f"lsof -ti:{port} | xargs kill -9")
        logger.info(f"Attempted to kill previous process on port {port}")
        time.sleep(1)  # Give the OS time to free up the port
    except Exception as e:
        logger.warning(f"Could not kill previous process: {e}")

def main():
    PORT = 50051
    HOST = 'localhost'
    MAX_RETRY = 3
    
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Try to bind to the port, with retries
    retry_count = 0
    while retry_count < MAX_RETRY:
        try:
            server_socket.bind((HOST, PORT))
            break
        except socket.error as e:
            if e.errno == 98:  # Address already in use
                retry_count += 1
                logger.warning(f"Port {PORT} is already in use (attempt {retry_count}/{MAX_RETRY})")
                
                if retry_count == 1:
                    # First try: attempt to kill the previous instance
                    logger.info("Attempting to kill the previous server instance...")
                    kill_previous_instance(PORT)
                elif retry_count == 2:
                    # Second try: wait a bit longer
                    logger.info("Waiting for port to become available...")
                    time.sleep(5)
                else:
                    # Last try: find a different port
                    try:
                        PORT = find_available_port(PORT + 1)
                        logger.info(f"Using alternative port: {PORT}")
                    except RuntimeError as e:
                        logger.error(str(e))
                        logger.error("Could not start server. Please ensure no other instances are running.")
                        return
            else:
                logger.error(f"Socket error: {e}")
                return
    
    try:
        server_socket.listen(1)
        logger.info(f"RIFE server started. Waiting for connections on {HOST}:{PORT}...")
        
        while True:
            try:
                # Accept connection
                client_socket, address = server_socket.accept()
                logger.info(f"Connection from {address}")
                
                try:
                    # Receive request
                    data = client_socket.recv(1024).decode()
                    
                    # Check for exit command
                    if data.strip() == "EXIT":
                        logger.info("Received exit command")
                        break
                    
                    # Parse request
                    try:
                        frame1_path, frame3_path, output_path, exp = data.strip().split('|')
                        exp = int(exp)
                    except ValueError:
                        # Try without exp (backward compatibility)
                        try:
                            frame1_path, frame3_path, output_path = data.strip().split('|')
                            exp = 1  # Default to single interpolation
                            print(f"Using default exp: {exp}")
                        except ValueError:
                            client_socket.sendall(b"ERROR: Invalid request format\n")
                            continue
                    
                    # Process request
                    response = process_request(frame1_path, frame3_path, output_path, exp)
                    client_socket.sendall(f"{response}\n".encode())
                    
                except Exception as e:
                    logger.error(f"Error handling client request: {e}")
                    client_socket.sendall(f"ERROR: {str(e)}\n".encode())
                finally:
                    client_socket.close()
                    
            except KeyboardInterrupt:
                logger.info("Server shutting down...")
                break
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                continue
                
    finally:
        server_socket.close()
        logger.info("Server stopped")

if __name__ == "__main__":
    main()