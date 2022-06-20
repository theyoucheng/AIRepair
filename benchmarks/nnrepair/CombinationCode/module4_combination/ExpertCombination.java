import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Combination Implementation
 */
public class ExpertCombination {

	/**
	 * Enumeration for the supported expert combination methods.
	 */
	public enum COMBINATION_METHOD {
		NAIVE, AVERAGE, FULL, PREC, CONF, VOTES, PVC, ORIG, ALL
	};

	/**
	 * Selects layer with maximum confidence.
	 * 
	 * @param values - values at last layer.
	 * @return final decision for original weights
	 */
	public static int selectLabelWithMaxConfidence(double[] values) {
		int largest = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] > values[largest]) {
				largest = i;
			}
		}
		return largest;
	}

	public static List<Integer> collectExpertClaims(int[] expertIDs, Map<Integer, double[]> result) {
		List<Integer> expertClaims = new ArrayList<>();
		for (int expertId : expertIDs) {
			int largest = selectLabelWithMaxConfidence(result.get(expertId));
			if (largest == expertId) {
				expertClaims.add(expertId);
			}
		}
		return expertClaims;
	}

	/**
	 * NAIVE Combination
	 * 
	 * Is there a unique expert who votes for itself? Otherwise return original
	 * choice.
	 * 
	 * @param expertClaims
	 * @param origLabel
	 * @param expertIDs
	 * @return
	 */
	public static int combineExpertsByNaive(List<Integer> expertClaims, int origLabel) {
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		} else {
			return origLabel;
		}
	}

	/**
	 * PREC Combination
	 * 
	 * @param expertClaims
	 * @param origLabel
	 * @param trainPrecision
	 * @param expertIDs
	 * @return
	 */
	public static Integer combineExpertsByPrecision(List<Integer> expertClaims, int origLabel,
			double[] trainPrecision) {

		/* Check whether this no or only one expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Select expert with highest precision on training data. */
		int maxExpertId = expertClaims.get(0);
		double maxPrecision = trainPrecision[maxExpertId];
		for (int expertId : expertClaims) {
			if (trainPrecision[expertId] > maxPrecision) {
				maxPrecision = trainPrecision[expertId];
				maxExpertId = expertId;
			}
		}
		return maxExpertId;
	}

	/**
	 * VOTES Combination
	 * 
	 * @param result
	 * @param expertClaims
	 * @param origLabel
	 * @param expertIDs
	 * @return
	 */
	public static int combineExpertsByVotes(Map<Integer, double[]> result, List<Integer> expertClaims, int origLabel,
			int[] expertIDs, int numberOfFinalLabels) {

		/* Check whether this no or only one expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Collect the vote by each expert, i.e., the label with maximum confidence. */
		int[] votes = new int[numberOfFinalLabels];
		for (int expertId : expertIDs) {
			int label = selectLabelWithMaxConfidence(result.get(expertId));
			votes[label]++;
		}

		/*
		 * Choose expert that votes for itself and that received most of the other
		 * votes.
		 */
		int maxVotedExpertId = expertClaims.get(0);
		for (int expertId : expertClaims) {
			if (votes[expertId] > votes[maxVotedExpertId]) {
				maxVotedExpertId = expertId;
			}
		}

		return maxVotedExpertId;
	}

	/**
	 * CONF Combination
	 * 
	 * @param expertClaims
	 * @param result
	 * @param origLabel
	 * @param expertIDs
	 * @return
	 */
	public static int combineExpertsByConfidence(Map<Integer, double[]> result, List<Integer> expertClaims,
			int origLabel) {

		/* Check whether this no or only expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Collect expert with highest confidence in its own label. */
		int highestConfidenceId = expertClaims.get(0);
		double highestConfidenceValue = result.get(highestConfidenceId)[highestConfidenceId];
		for (int expertId : expertClaims) {
			double confidenceValue = result.get(expertId)[expertId];
			if (confidenceValue > highestConfidenceValue) {
				highestConfidenceValue = confidenceValue;
				highestConfidenceId = expertId;
			}
		}

		return highestConfidenceId;
	}

	/**
	 * PVC Combination
	 * 
	 * @param result
	 * @param expertClaims
	 * @param origLabel
	 * @param trainPrecision
	 * @param expertIDs
	 * @return
	 */
	public static Integer combineExpertsByPVC(Map<Integer, double[]> result, List<Integer> expertClaims, int origLabel,
			double[] trainPrecision, int[] expertIDs, int numberOfFinalLabels) {

		/* Check whether this no or only one expert claiming its label. */
		if (expertClaims.isEmpty()) {
			return origLabel;
		}
		if (expertClaims.size() == 1) {
			return expertClaims.get(0);
		}

		/* Initialize scores. */
		Map<Integer, Integer> scorePerExpert = new HashMap<>();
		for (int expertId : expertClaims) {
			scorePerExpert.put(expertId, 0);
		}

		/* Calculate scores */
		int labelByPrecision = combineExpertsByPrecision(expertClaims, origLabel, trainPrecision);
		scorePerExpert.put(labelByPrecision, scorePerExpert.get(labelByPrecision) + 1);

		int labelByVotes = combineExpertsByVotes(result, expertClaims, origLabel, expertIDs, numberOfFinalLabels);
		scorePerExpert.put(labelByVotes, scorePerExpert.get(labelByVotes) + 1);

		int labelByConfidence = combineExpertsByConfidence(result, expertClaims, origLabel);
		scorePerExpert.put(labelByConfidence, scorePerExpert.get(labelByConfidence) + 1);

		/* Pick maximum score. */
		int maxScore = -1;
		int maxScoreExpert = -1;
		for (Entry<Integer, Integer> expertScorePair : scorePerExpert.entrySet()) {
			if (expertScorePair.getValue() > maxScore) {
				maxScore = expertScorePair.getValue();
				maxScoreExpert = expertScorePair.getKey();
			}
		}

		return maxScoreExpert;
	}

	public static Map<COMBINATION_METHOD, Integer> combineExperts(COMBINATION_METHOD combMethod,
			Map<Integer, double[]> result, int origLabel, double[] trainPrecision, int[] expertIDs, boolean optimized,
			int numberOfFinalLabels) {
		Map<COMBINATION_METHOD, Integer> combinedResults = new HashMap<>();

		if (combMethod.equals(COMBINATION_METHOD.ORIG) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.ORIG, origLabel);
		}

		List<Integer> expertClaims = collectExpertClaims(expertIDs, result);

		if (!optimized) {
			if (combMethod.equals(COMBINATION_METHOD.AVERAGE) || combMethod.equals(COMBINATION_METHOD.ALL)) {
				combinedResults.put(COMBINATION_METHOD.AVERAGE, selectLabelWithMaxConfidence(result.get(11)));
			}

			if (combMethod.equals(COMBINATION_METHOD.FULL) || combMethod.equals(COMBINATION_METHOD.ALL)) {
				combinedResults.put(COMBINATION_METHOD.FULL, selectLabelWithMaxConfidence(result.get(10)));
			}
		}

		if (combMethod.equals(COMBINATION_METHOD.NAIVE) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.NAIVE, combineExpertsByNaive(expertClaims, origLabel));
		}

		if (combMethod.equals(COMBINATION_METHOD.PREC) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			if (trainPrecision.length > 0) {
				combinedResults.put(COMBINATION_METHOD.PREC,
						combineExpertsByPrecision(expertClaims, origLabel, trainPrecision));	
			}
		}

		if (combMethod.equals(COMBINATION_METHOD.CONF) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.CONF, combineExpertsByConfidence(result, expertClaims, origLabel));
		}

		if (combMethod.equals(COMBINATION_METHOD.VOTES) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			combinedResults.put(COMBINATION_METHOD.VOTES,
					combineExpertsByVotes(result, expertClaims, origLabel, expertIDs, numberOfFinalLabels));
		}

		if (combMethod.equals(COMBINATION_METHOD.PVC) || combMethod.equals(COMBINATION_METHOD.ALL)) {
			if (trainPrecision.length > 0) {
				combinedResults.put(COMBINATION_METHOD.PVC, combineExpertsByPVC(result, expertClaims, origLabel,
						trainPrecision, expertIDs, numberOfFinalLabels));
			}
		}

		return combinedResults;
	}

}
