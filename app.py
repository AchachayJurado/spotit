import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import json
import base64
from pose_detector import PoseDetector
from pose_matcher import PoseMatcher
from depth_detector import DepthDetector
import os
from PIL import Image

class PoseGame:
    def __init__(self):
        # Load settings
        self.settings_file = os.path.join(os.path.dirname(__file__), 'settings.json')
        with open(self.settings_file, 'r') as f:
            self.settings = json.load(f)
        
        # Initialize components with settings
        self.detector = PoseDetector(confidence_threshold=self.settings['pose_detector']['confidence_threshold'])
        self.matcher = PoseMatcher(
            similarity_threshold=self.settings['pose_matcher']['similarity_threshold'],
            confidence_threshold=self.settings['pose_detector']['confidence_threshold']
        )
        self.depth_detector = DepthDetector()
        
        # Game settings
        self.game_duration = self.settings['game']['duration_seconds']
        self.pose_hold_time = self.settings['game']['pose_hold_time_seconds']
        self.pose_hold_start = None
        self.reference_poses = self._load_reference_poses()
        self.fps_history = []  # Track FPS
        if not self.reference_poses:
            st.error("""
            No reference poses found! Please add some .jpg or .png images to the 'static/poses' directory.
            Check the README.md file in that directory for guidelines on suitable images.
            """)
            st.stop()
        self.current_pose_idx = 0
        
        # Load or create player data file
        self.player_data_file = os.path.join(os.path.dirname(__file__), 'player_data.json')
        if os.path.exists(self.player_data_file):
            with open(self.player_data_file, 'r') as f:
                self.player_data = json.load(f)
        else:
            self.player_data = []

    def _save_player_data(self, name, email, phone, score):
        """Save player data to JSON file"""
        # Check if player exists and update only if new score is better
        existing_player = None
        for player in self.player_data:
            if player['name'] == name and player['email'] == email:
                if player['score'] >= score:  # Don't update if current score is lower
                    return
                existing_player = player
                break
        
        if existing_player:
            existing_player['score'] = score
            existing_player['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            existing_player['phone'] = phone  # Update phone if changed
        else:
            self.player_data.append({
                'name': name,
                'email': email,
                'phone': phone,
                'score': score,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        with open(self.player_data_file, 'w') as f:
            json.dump(self.player_data, f, indent=2)

    def _resize_image(self, img, max_height=400):
        """Resize image to fit screen while maintaining aspect ratio"""
        height, width = img.shape[:2]
        if height > max_height:
            ratio = max_height / height
            new_width = int(width * ratio)
            img = cv2.resize(img, (new_width, max_height))
        return img
    
    def _resize_frame(self, frame, max_height=400):
        """Resize frame to fit screen while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        if height > max_height:
            ratio = max_height / height
            new_width = int(width * ratio)
            frame = cv2.resize(frame, (new_width, max_height))
        return frame
        
    def _load_reference_poses(self):
        """Load and process reference poses"""
        poses_dir = os.path.join(os.path.dirname(__file__), 'static', 'spotit')
        pose_files = [f for f in os.listdir(poses_dir) if f.endswith(('.jpg', '.JPG'))]
        reference_poses = []
        
        for pose_file in pose_files:
            img_path = os.path.join(poses_dir, pose_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = self._resize_image(img)
                keypoints = self.detector.detect_pose(img)
                if keypoints is not None:
                    reference_poses.append({
                        'image': img,
                        'keypoints': keypoints,
                        'name': os.path.splitext(pose_file)[0]
                    })
        
        return reference_poses

    def _update_next_pose(self, ref_pose_placeholder):
        """Update to next pose"""
        st.session_state.score += 1
        st.session_state.current_pose_idx = (st.session_state.current_pose_idx + 1) % len(self.reference_poses)
        self.pose_hold_start = None
        time.sleep(0.5)  # Add delay for visibility
        ref_pose_img = self.reference_poses[st.session_state.current_pose_idx]['image']
        ref_pose_placeholder.image(ref_pose_img, channels="BGR")

    def run(self):
        # Set page theme
        st.markdown("""
            <style>
                .stApp {
                    background-color: #003366;
                    color: white;
                }
                .stButton>button {
                    background-color: #003366;
                    color: white;
                    border: 1px solid #004080;
                }
                .stButton>button:hover {
                    background-color: #004080;
                    border: 1px solid #0059b3;
                }
                .stTextInput>div>div>input {
                    background-color: #002b57;
                    color: white !important;
                    border: 1px solid #004080;
                }
                .stTextInput>label {
                    color: white !important;
                }
                .stTextInput>div>div::placeholder {
                    color: rgba(255,255,255,0.7) !important;
                }
                .stMarkdown {
                    color: white;
                }
                h1, h2, h3 {
                    color: white !important;
                }
                .dataframe {
                    background-color: #002b57 !important;
                }
                .dataframe th {
                    background-color: #003366 !important;
                    color: white !important;
                    font-weight: bold !important;
                }
                .dataframe td {
                    background-color: #002b57 !important;
                    color: white !important;
                }
                .dataframe tbody tr th {
                    color: white !important;
                }
                .dataframe tbody td {
                    color: white !important;
                }
                .dataframe * {
                    color: white !important;
                    text-align: left !important;
                }
                .highlight {
                    background-color: #004080 !important;
                }
                .stAlert {
                    background-color: #002b57 !important;
                    color: white !important;
                }
                .stAlert > div {
                    color: white !important;
                }
                div[data-testid="stTable"] {
                    color: white !important;
                }
                div[data-testid="stTable"] * {
                    color: white !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Center the title
        st.markdown("<h1 style='text-align: center;'>🛎 Spot AIt! 🛎</h1>", unsafe_allow_html=True)
        
        # Initialize session state
        if 'game_active' not in st.session_state:
            st.session_state.game_active = False
            st.session_state.show_leaderboard = False
            st.session_state.name_entered = False
            st.session_state.score = 0
            st.session_state.high_scores = []
            st.session_state.current_pose_idx = 0
            st.session_state.player_name = ""
            st.session_state.player_email = ""
            st.session_state.player_phone = ""
            st.session_state.latest_player = None
        
        # Name and email entry page
        if not st.session_state.name_entered:
            # Create a container for the entire content to control layout
            container = st.container()
            
            # Add content to the container
            with container:
                # Welcome message and user entry section
                st.markdown("<h3 style='text-align: center;'>AI vs HI</h3>", unsafe_allow_html=True)                
                # Center the input fields
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    player_name = st.text_input("Human Name", key="name_input")
                    player_email = st.text_input("AI Name", key="email_input")

                    if st.button("Continue") and player_name.strip() :
                        st.session_state.player_name = player_name.strip()
                        st.session_state.player_email = player_email.strip()
                        st.session_state.name_entered = True
                        st.rerun()
                
                
                # Add small space before logos
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                # Display logos at the bottom in a single row
                logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
                with logo_col2:
                    # Create two columns for the logos
                    left_logo,middle_logo, right_logo = st.columns(3)
                    with left_logo:
                        st.image(os.path.join(os.path.dirname(__file__), "static", "logos/logo_latent.png"), width=290)
                    with middle_logo:
                        st.image(os.path.join(os.path.dirname(__file__), "static", "logos/nvidia.png"), width=330)
                    with right_logo:
                        st.image(os.path.join(os.path.dirname(__file__), "static", "logos/Dell_Technologies-Logo.wine_resized.png"), width=330)
            
            return

        # Sidebar navigation
        with st.sidebar:
            if st.button("🏠 Home"):
                st.session_state.game_active = False
                st.session_state.show_leaderboard = False
                st.session_state.name_entered = False
                st.rerun()
            
            if not st.session_state.game_active:
                if st.button("🎮 Start Game"):
                    st.session_state.game_active = True
                    st.session_state.show_leaderboard = False
                    st.session_state.start_time = time.time()
                    st.session_state.score = 0
                    st.session_state.current_pose_idx = 0
                    self.pose_hold_start = None
                    self.fps_history = []  # Reset FPS history
                    # Shuffle reference poses
                    self.reference_poses = np.random.permutation(self.reference_poses).tolist()
            
            if st.button("🏆 Leaderboard"):
                st.session_state.show_leaderboard = True
                st.session_state.game_active = False
        
        if st.session_state.show_leaderboard:
            self._show_leaderboard(st.session_state.latest_player)
        elif st.session_state.game_active:
            self._run_game_loop()
        else:
            # Add space before instructions
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Display centered instructions
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h3>Rules</h3>
                    <br>
                    <p>Be the fastest to name the only common symbol between 2 cards</p>
                    <br>
                </div>
            """, unsafe_allow_html=True)
            
            # Add space before Play button
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Add centered Play button and audio container
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                # Add containers
                button_container = st.empty()
                countdown_container = st.empty()
                
                # Show Play Game button
                if button_container.button("🎮 Play Game", use_container_width=True):
                    # Clear the button immediately
                    button_container.empty()
                    
                    # Play audio with custom styled player
                    audio_path = os.path.join(os.path.dirname(__file__), "static", "audio", "Dell_audio.m4a")
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    
                    st.markdown(f"""
                        <style>
                            audio {{
                                width: 100%;
                                height: 40px;
                                background-color: #002b57;
                                border-radius: 5px;
                                margin-bottom: 10px;
                            }}
                        </style>
                        <audio autoplay controls style="display: none;">
                            <source src="data:audio/mp4;base64,{audio_b64}" type="audio/mp4">
                        </audio>
                    """, unsafe_allow_html=True)
                    
                    # Show "Get Ready!" message
                    countdown_container.markdown("<h2 style='text-align: center;'>Get Ready!</h2>", unsafe_allow_html=True)
                    
                    # Progress bar for audio intro
                    progress_text = "Audio intro in progress..."
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.18)  # Total ~19 seconds
                    progress_bar.empty()
                    
                    # Display countdown synchronized with audio
                    for i in range(3, 0, -1):
                        countdown_container.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{i}</h1>", unsafe_allow_html=True)
                        time.sleep(1)
                    countdown_container.empty()
                    
                    # Small delay after countdown before starting game
                    time.sleep(0.5)
                    
                    st.session_state.game_active = True
                    st.session_state.show_leaderboard = False
                    st.session_state.start_time = time.time()
                    st.session_state.score = 0
                    st.session_state.current_pose_idx = 0
                    self.pose_hold_start = None
                    self.fps_history = []  # Reset FPS history
                    # Shuffle reference poses
                    self.reference_poses = np.random.permutation(self.reference_poses).tolist()
                    st.rerun()

    def _run_game_loop(self):
        # Create container for game UI
        game_container = st.container()
        
        # Create columns for displays
        with game_container:
            # Setup placeholders for stats
            stats_col1, stats_col2, similarity_col, fps_col = st.columns([1, 1, 1, 1])
            with stats_col1:
                time_placeholder = st.empty()
            with stats_col2:
                score_placeholder = st.empty()
            with similarity_col:
                similarity_placeholder = st.empty()
            with fps_col:
                fps_placeholder = st.empty()
            
            # Status placeholder at the top
            status_placeholder = st.empty()
            
            # Create three equal columns for main frames
            col1, col2, col3 = st.columns(3)
            
            # Reference pose
            with col1:
                st.subheader("Reference Pose")
                ref_pose_placeholder = st.empty()
                ref_pose_img = self.reference_poses[st.session_state.current_pose_idx]['image']
                ref_pose_placeholder.image(ref_pose_img, channels="BGR")
            
            # Camera feed
            with col2:
                st.subheader("Your Pose")
                camera_placeholder = st.empty()
            
            # Depth map and raw feed in third column
            with col3:
                st.subheader("Depth Map")
                depth_placeholder = st.empty()
                st.subheader("Raw Camera Feed")
                raw_camera_placeholder = st.empty()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Game loop
        while st.session_state.game_active:
            loop_start_time = time.time()
            
            elapsed_time = int(time.time() - st.session_state.start_time)
            remaining_time = max(0, self.game_duration - elapsed_time)
            
            # Update timer and score
            time_placeholder.markdown(f"### ⏱️ Time: {remaining_time}s")
            score_placeholder.markdown(f"### 🎯 Score: {st.session_state.score}")
            
            # Check if game should end
            if remaining_time == 0:
                self._end_game()
                break
            
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            try:
                # Get depth map and person box
                depth_map, person_box = self.depth_detector.detect_depth(frame)
                
                # Crop frame to person
                if person_box is not None:
                    cropped_frame = self.depth_detector.crop_frame(frame, person_box)
                else:
                    cropped_frame = frame
                
                # Resize frames to fit screen
                cropped_frame = self._resize_frame(cropped_frame)
                colored_depth = self._resize_frame(self.depth_detector.get_colored_depth_map(depth_map))
                
                # Detect pose on cropped frame
                detected_keypoints = self.detector.detect_pose(cropped_frame)
                
                # Update depth map
                depth_placeholder.image(colored_depth)
                
                # Get reference keypoints
                ref_keypoints = self.reference_poses[st.session_state.current_pose_idx]['keypoints']
                
                if detected_keypoints is not None and len(detected_keypoints) >= len(ref_keypoints):  # Only need matching number of keypoints
                    # Draw pose on frame
                    cropped_frame = self.detector.draw_pose(cropped_frame, detected_keypoints)
                    
                    # Check pose similarity
                    is_matched, similarity = self.matcher.is_pose_matched(
                        detected_keypoints, ref_keypoints
                    )
                    
                    # Display similarity score outside frame
                    similarity_placeholder.markdown(
                        f"### {'🟢' if is_matched else '🔴'} Similarity: {similarity:.2f}"
                    )
                    
                    # Update pose lock status
                    if is_matched:
                        if self.pose_hold_start is None:
                            self.pose_hold_start = time.time()
                            hold_duration = 0
                        else:
                            hold_duration = time.time() - self.pose_hold_start
                        
                        if hold_duration >= self.pose_hold_time:
                            status_placeholder.markdown("### 🔒 Pose: Locked!")
                            self._update_next_pose(ref_pose_placeholder)
                        else:
                            remaining = self.pose_hold_time - hold_duration
                            status_placeholder.markdown(f"### 🔄 Hold Pose: {remaining:.1f}s")
                    else:
                        self.pose_hold_start = None
                        status_placeholder.markdown("### 🔓 Pose: None")
                else:
                    status_placeholder.markdown("### 🔍 Waiting for full pose detection...")
                    similarity_placeholder.empty()
                
                # Update frame displays
                camera_placeholder.image(cropped_frame, channels="BGR")
                raw_camera_placeholder.image(frame, channels="BGR")
                
                # Calculate and display FPS
                loop_time = time.time() - loop_start_time
                fps = 1.0 / loop_time
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:  # Keep last 30 frames for average
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                fps_placeholder.markdown(f"### 🔄 FPS: {avg_fps:.1f}")
                    
            except Exception as e:
                print(f"Error in game loop: {e}")
                status_placeholder.markdown("### ⚠️ Processing error")
                camera_placeholder.image(frame, channels="BGR")
                similarity_placeholder.empty()
        
        # Release webcam
        cap.release()

    def _show_leaderboard(self, highlight_player=None):
        """Display the leaderboard page"""
        st.subheader("🏆 Leaderboard")
        
        if self.player_data:
            # Convert player data to DataFrame
            scores_df = pd.DataFrame(self.player_data)
            scores_df = scores_df.sort_values('score', ascending=False)
            
            # Format player names
            scores_df['name'] = scores_df['name'].apply(lambda x: f"**{x}**")
            
            # Add rank column
            scores_df['rank'] = range(1, len(scores_df) + 1)
            scores_df['rank'] = scores_df['rank'].apply(
                lambda x: f"🥇 {x}" if x == 1 else f"🥈 {x}" if x == 2 else f"🥉 {x}" if x == 3 else f"{x}"
            )
            
            # Highlight current player
            if highlight_player:
                scores_df['name'] = scores_df.apply(
                    lambda row: f"{row['name']} 🔴" if row['name'].replace('**', '') == highlight_player else row['name'],
                    axis=1
                )
            
            # Reorder columns
            scores_df = scores_df[['rank', 'name', 'email', 'score', 'timestamp']]
            scores_df.columns = ['Rank', 'Player', 'Email', 'Score', 'Date']
            
            # Style the DataFrame
            styled_df = scores_df.style.apply(
                lambda x: ['background-color: #004080' if x.name == scores_df[scores_df['Player'].str.contains('🔴', na=False)].index[0] else '' for i in x],
                axis=1
            ) if highlight_player and any(scores_df['Player'].str.contains('🔴', na=False)) else scores_df
            
            st.table(styled_df)
        else:
            st.info("No scores yet! Play a game to set a record.")
            
    def _end_game(self):
        st.session_state.game_active = False
        st.session_state.latest_player = st.session_state.player_name
        
        # Save player data to JSON
        self._save_player_data(
            st.session_state.player_name,
            st.session_state.player_email,
            st.session_state.player_phone,
            st.session_state.score
        )
        
        # Show game over message
        st.success(f"""
        ### 🎮 Game Over!
        Player: {st.session_state.player_name}
        Final Score: {st.session_state.score}
        Average FPS: {sum(self.fps_history) / len(self.fps_history):.1f}
        """)
        
        if st.button("🏆 View Leaderboard"):
            st.session_state.show_leaderboard = True
            st.rerun()
        
        st.markdown("Click '🎮 Start Game' to play again!")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Spot AIt!",
        page_icon="🛎",
        layout="wide"
    )
    
    game = PoseGame()
    game.run()
