def onStart():
	import numpy as np
	import matplotlib.pyplot as plt
	import librosa
	from scipy.ndimage.filters import maximum_filter
	from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure
	import collections
	
	print('START IMPORT')
	
	# ----------------------
	# Load Tracks
	# ----------------------
	print('LOADING TRACKS')
	
	content_track, content_sr = librosa.load('dfc_int_assets/TDAudio_audio1.wav', sr=48000)
	print('CONTENT SR: ', content_sr)
	
	playback_track, playback_sr = librosa.load('dfc_int_assets/TDAudio_audio2.wav', sr=48000)
	
	# -----------------------
	# Create Power Spectrogram
	# -----------------------
	print('CREATING POWER SPECTROGRAM')
	
	content_sg = np.abs(librosa.stft(content_track, n_fft=512))
	# print('SG SHAPE: ', content_sg.shape)
	playback_sg = np.abs(librosa.stft(playback_track, n_fft=512))
	
	print('CREATING POWER SPECTROGRAM FINISHED')
	
	# -----------------------
	# Find Local Peaks
	# -----------------------
	def get_2D_peaks(arr2D: np.array, plot: bool = True, amp_min: int = 10):
	    """
	    Extract maximum peaks from the spectogram matrix (arr2D).
	    :param arr2D: matrix representing the spectogram.
	    :param plot: for plotting the results.
	    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
	    :return: a list composed by a list of frequencies and times.
	    """
	    # Original code from the repo is using a morphology mask that does not consider diagonal elements
	    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
	    # is to change from the current diamond figure to a just a normal square one:
	    #       F   T   F           T   T   T
	    #       T   T   T   ==>     T   T   T
	    #       F   T   F           T   T   T
	    # In my local tests time performance of the square mask was ~3 times faster
	    # respect to the diamond one, without hurting accuracy of the predictions.
	    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
	    # That being said, we generate the mask by using the following function
	    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
	    struct = generate_binary_structure(2, 2)
	
	    #  And then we apply dilation using the following function
	    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
	    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
	    #  change it by the following code:
	    #  neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
	    neighborhood = iterate_structure(struct, 10)
	
	    # find local maxima using our filter mask
	    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
	
	    # Applying erosion, the dejavu documentation does not talk about this step.
	    background = (arr2D == 0)
	    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
	
	    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
	    detected_peaks = local_max != eroded_background
	
	    # extract peaks
	    amps = arr2D[detected_peaks]
	    freqs, times = np.where(detected_peaks)
	
	    # filter peaks
	    amps = amps.flatten()
	
	    # get indices for frequency and time
	    filter_idxs = np.where(amps > amp_min)
	
	    freqs_filter = freqs[filter_idxs]
	    times_filter = times[filter_idxs]
	
	    # print(amps > amp_min)
	
	    return detected_peaks
	    #return np.array(list(zip(times_filter, freqs_filter)))
	
	# -----------------------
	# Align Constellation Maps
	# -----------------------
	
	print('GENERATE CMs')
	
	scores = {}
	bin_size = 200
	print('BIN SIZE', bin_size)
	content_cm = get_2D_peaks(content_sg)  # 2d constellation map for the content audio
	playback_cm = get_2D_peaks(playback_sg)  # 2d constellation map for the playback audio
	# we assume that dim 0 is the time frame
	# and dim 1 is the frequency bin
	# both CMs contains only 0 or 1
	
	print('COUNT FRAMES')
	
	content_cm = np.transpose(content_cm)
	content_frames = content_cm.shape[0]
	print('Content Frames', content_frames,content_cm.shape)
	
	playback_cm = np.transpose(playback_cm)
	playback_frames = playback_cm.shape[0]
	print('Playback Frames', playback_frames, playback_cm.shape)
	
	# Match against the middle of the playback (to guarantee overlap to test)
	excerpt_start = playback_frames // 2 - bin_size // 2
	playback_cm_excerpt = playback_cm[excerpt_start:excerpt_start + bin_size]
	
	print('POINTS IN CONSTELLATION', np.sum(playback_cm_excerpt))
	
	# Slide content excerpt against
	for offset in range(content_frames - bin_size):
	    # print(offset, offset + bin_size, len(content_cm))
	    content_cm_excerpt = content_cm[offset:offset + bin_size]
	    score = np.sum(np.multiply(playback_cm_excerpt, content_cm_excerpt))
	    scores[offset - excerpt_start] = score
	
	# -----------------------
	# Return Max Count and Return Position
	# -----------------------
	
	print('MOST COMMON FRAME', collections.Counter(scores).most_common(10))
	
	key, _ = max(scores.items(), key=lambda kv: kv[1])
	print(key)
	
	time_offset = librosa.frames_to_time(key, content_sr, hop_length=128, n_fft=None)
	
	print('TIME OFFSET DETECTED', time_offset)
	
	print('SCRIPT FINISHED')

	return
