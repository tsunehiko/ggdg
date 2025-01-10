import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import features.feature_sets.network.JITSPatterNetFeatureSet;
import game.Game;
import gnu.trove.list.array.TIntArrayList;
import main.CommandLineArgParse;
import main.CommandLineArgParse.ArgOption;
import main.CommandLineArgParse.OptionTypes;
import main.collections.ListUtils;
import other.AI;
import other.GameLoader;
import other.RankUtils;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;
import utils.AIFactory;
import utils.experiments.ResultsSummary;

import supplementary.experiments.eval.ParallelEvalMultiGamesMultiAgents;

/**
 * Implementation of an experiment that evaluates multiple agents across multiple games.
 * This is the single-threaded version.
 * 
 * @author 
 */
public class ComputeMultiAgents extends ParallelEvalMultiGamesMultiAgents 
{
	
	//-------------------------------------------------------------------------
	
	protected List<String> jsonFiles;

	/** 
	 * Whether to create a small GUI that can be used to manually interrupt training run. 
	 * False by default. 
	 */
	protected boolean useGUI;

	/** Max wall time in minutes (or -1 for no limit) */
	protected int maxWallTime;
	
	/**
	 * Starts the experiment in a single-threaded manner.
	 */
	public void startExperiment()
	{
		final long startTime = System.currentTimeMillis();
		final Random random = new Random();
		
		try
		{
			for (final String jsonFile : jsonFiles)
			{
				// Load a batch of trials from JSON
				final TrialsBatchToRun trialsBatch = TrialsBatchToRun.fromJson(jsonFile);
				
				if (trialsBatch == null)
				{
					System.err.println("Failed to load trials batch from: " + jsonFile);
					continue;
				}
				
				// Load game for this batch
				final Game game;
				
				if (trialsBatch.treatGameNameAsFilepath)
				{
					if (trialsBatch.ruleset != null && !trialsBatch.ruleset.equals(""))
						game = GameLoader.loadGameFromFile(new File(trialsBatch.gameName), trialsBatch.ruleset);
					else
						game = GameLoader.loadGameFromFile(new File(trialsBatch.gameName), new ArrayList<String>());	// TODO add support for options
				}
				else
				{
					if (trialsBatch.ruleset != null && !trialsBatch.ruleset.equals(""))
						game = GameLoader.loadGameFromName(trialsBatch.gameName, trialsBatch.ruleset);
					else
						game = GameLoader.loadGameFromName(trialsBatch.gameName, new ArrayList<String>());	// TODO add support for options
				}
				
				// Clear some unnecessary memory
				game.description().setParseTree(null);
				game.description().setExpanded(null);
				
				final int numPlayers = game.players().count();
				
				if (trialsBatch.agentStrings.length != numPlayers)
				{
					System.err.println
					(
						"Expected " + numPlayers + 
						" agents, but received list of " + trialsBatch.agentStrings.length + 
						" agents. Aborting set of games."
					);
					break;
				}
				
                // Set game max move limit
                // System.out.println("Game: " + trialsBatch.gameName + ", LengthCap: " + trialsBatch.gameLengthCap);
				if (trialsBatch.gameLengthCap >= 0)
					game.setMaxMoveLimit(Math.min(trialsBatch.gameLengthCap, game.getMaxMoveLimit()));
                // System.out.println("Game: " + trialsBatch.gameName + ", MaxMoves: " + game.getMaxMoveLimit());
				
				// Permutations of agents list, to rotate through
				final List<TIntArrayList> aiListPermutations;
				if (numPlayers <= 5)
				{
					// Compute all possible permutations of indices for the list of AIs
					aiListPermutations = ListUtils.generatePermutations(
							TIntArrayList.wrap(IntStream.range(0, numPlayers).toArray()));
					
					Collections.shuffle(aiListPermutations, random);
				}
				else
				{
					// Randomly generate some permutations of indices for the list of AIs
					aiListPermutations = ListUtils.samplePermutations(TIntArrayList.wrap(IntStream.range(0, numPlayers).toArray()), 120);
				}
				
				try
				{
					// Do a warming up (just a few seconds anyway, so sequential execution is fine)
					final Trial trial = new Trial(game);
					final Context context = new Context(game, trial);
					long stopAt = 0L;
					final long start = System.nanoTime();
					final double abortAt = start + trialsBatch.warmingUpSecs * 1_000_000_000.0;
					while (stopAt < abortAt)
					{
						game.start(context);
						game.playout(context, null, 1.0, null, -1, -1, random);
						stopAt = System.nanoTime();
					}
					System.gc();
				}
				catch (final Exception e)
				{
					System.err.println("Crash during warmup for game: " + trialsBatch.gameName);
					e.printStackTrace();
					continue;
				}
				
				// Prepare results writing (ResultsSummary is thread-safe)
				final List<String> agentStrings = new ArrayList<String>();
				for (final String agentString : trialsBatch.agentStrings)
				{
					agentStrings.add(AIFactory.createAI(agentString).friendlyName());
				}
				final ResultsSummary resultsSummary = new ResultsSummary(game, agentStrings);
				
				// System.out.println("Processing game: " + trialsBatch.gameName);
				// System.out.println("Ruleset: " + trialsBatch.ruleset);
				// for (final String agentString : trialsBatch.agentStrings)
				// {
				// 	System.out.println("Agent: " + agentString);
				// }
				
				for (int trialCounter = 0; trialCounter < trialsBatch.numTrials; ++trialCounter)
				{
					try
					{								
						// Compute list of AIs to use for this trial (we rotate every trial)
						final List<AI> currentAIList = new ArrayList<AI>(numPlayers + 1);
						final int currentAIsPermutation = trialCounter % aiListPermutations.size();
						
						final TIntArrayList currentPlayersPermutation = aiListPermutations.get(currentAIsPermutation);
						currentAIList.add(null); // 0 index not used

						for (int i = 0; i < currentPlayersPermutation.size(); ++i)
						{
							currentAIList.add
							(
								AIFactory.createAI(trialsBatch.agentStrings[currentPlayersPermutation.getQuick(i) % numPlayers])
							);
						}

						// Play a game
						final Trial trial = new Trial(game);
						final Context context = new Context(game, trial);
						game.start(context);

						for (int p = 1; p < currentAIList.size(); ++p)
						{
							currentAIList.get(p).initAI(game, p);
						}
						
						final Model model = context.model();

						while (!context.trial().over())
						{
							model.startNewStep
							(
								context, currentAIList, trialsBatch.thinkingTime, trialsBatch.iterationLimit, 
								-1, 0.0
							);
						}
						
						// Close AIs
						for (int p = 1; p < currentAIList.size(); ++p)
						{
							currentAIList.get(p).closeAI();
						}

						// Record results
						if (context.trial().over())
						{
							final double[] utilities = RankUtils.agentUtilities(context);
							final int numMovesPlayed = context.trial().numMoves() - context.trial().numInitialPlacementMoves();
							final int[] agentPermutation = new int[currentPlayersPermutation.size() + 1];
							currentPlayersPermutation.toArray(agentPermutation, 0, 1, currentPlayersPermutation.size());
							
							resultsSummary.recordResults(agentPermutation, utilities, numMovesPlayed);
						}
					}
					catch (final Exception e)
					{
						e.printStackTrace();
					}
				}

				// After all trials, write results
				// System.out.println("Finished all trials for game: " + trialsBatch.gameName);
				if (trialsBatch.outDir != null)
				{
					if (trialsBatch.outputSummary)
					{
						final File outFile = new File(trialsBatch.outDir + "/results.txt");
						outFile.getParentFile().mkdirs();
						try (final PrintWriter writer = new PrintWriter(outFile, "UTF-8"))
						{
							writer.write(resultsSummary.generateIntermediateSummary());
						}
						catch (final FileNotFoundException | UnsupportedEncodingException e)
						{
							e.printStackTrace();
						}
					}
					
					if (trialsBatch.outputAlphaRankData)
					{
						final File outFile = new File(trialsBatch.outDir + "/alpha_rank_data.csv");
						outFile.getParentFile().mkdirs();
						resultsSummary.writeAlphaRankData(outFile);
					}
					
					if (trialsBatch.outputRawResults)
					{
						// System.out.println("writing raw results to " + trialsBatch.outDir + "/raw_results.csv");
						final File outFile = new File(trialsBatch.outDir + "/raw_results.csv");
						outFile.getParentFile().mkdirs();
						resultsSummary.writeRawResults(outFile);
					}
				}
			}
			
			// Handle max wall time
			if (maxWallTime > 0)
			{
				final long elapsedTime = System.currentTimeMillis() - startTime;
				final long maxWallTimeMillis = maxWallTime * 60 * 1000L;
				if (elapsedTime >= maxWallTimeMillis)
				{
					System.out.println("Max wall time reached. Stopping experiment.");
					return;
				}
			}
		}
		catch (final Exception e) {
			e.printStackTrace();
		}
	}

	//-------------------------------------------------------------------------
	
	/**
	 * A single batch of trials (between a specific set of agents for a single game)
	 * that we wish to run.
	 * 
	 * @author 
	 */
	public static class TrialsBatchToRun
	{
		
		protected final String gameName;
		protected final String ruleset;
		protected final int numTrials;
		protected final int gameLengthCap;
		protected final double thinkingTime;
		protected final int iterationLimit;
		protected final int warmingUpSecs;
		protected final String outDir;
		protected final String[] agentStrings;
		protected final boolean outputSummary;
		protected final boolean outputAlphaRankData;
		protected final boolean outputRawResults;
		protected final boolean treatGameNameAsFilepath;
		
		/**
		 * Constructor
		 * 
		 * @param gameName
		 * @param ruleset
		 * @param numTrials
		 * @param gameLengthCap
		 * @param thinkingTime
		 * @param iterationLimit
		 * @param warmingUpSecs
		 * @param outDir
		 * @param agentStrings
		 * @param outputSummary
		 * @param outputAlphaRankData
		 * @param outputRawResults
		 * @param treatGameNameAsFilepath
		 */
		public TrialsBatchToRun
		(
			final String gameName, final String ruleset, final int numTrials, final int gameLengthCap, 
			final double thinkingTime, final int iterationLimit, final int warmingUpSecs, final String outDir, 
			final String[] agentStrings, final boolean outputSummary, final boolean outputAlphaRankData,
			final boolean outputRawResults, final boolean treatGameNameAsFilepath
		) 
		{
			this.gameName = gameName;
			this.ruleset = ruleset;
			this.numTrials = numTrials;
			this.gameLengthCap = gameLengthCap;
			this.thinkingTime = thinkingTime;
			this.iterationLimit = iterationLimit;
			this.warmingUpSecs = warmingUpSecs;
			this.outDir = outDir;
			this.agentStrings = agentStrings;
			this.outputSummary = outputSummary;
			this.outputAlphaRankData = outputAlphaRankData;
			this.outputRawResults = outputRawResults;
			this.treatGameNameAsFilepath = treatGameNameAsFilepath;
		}
		
		public void toJson(final String jsonFilepath)
		{
			BufferedWriter bw = null;
			try
			{
				final File file = new File(jsonFilepath);
				file.getParentFile().mkdirs();
				if (!file.exists())
					file.createNewFile();

				final JSONObject json = new JSONObject();
				
				json.put("gameName", gameName);
				json.put("ruleset", ruleset);
				json.put("numTrials", numTrials);
				json.put("gameLengthCap", gameLengthCap);
				json.put("thinkingTime", thinkingTime);
				json.put("iterationLimit", iterationLimit);
				json.put("warmingUpSecs", warmingUpSecs);
				json.put("outDir", outDir);
				final JSONArray agentStringsJsonArray = new JSONArray(Arrays.asList(agentStrings));
				json.put("agentStrings", agentStringsJsonArray);
				json.put("outputSummary", outputSummary);
				json.put("outputAlphaRankData", outputAlphaRankData);
				json.put("outputRawResults", outputRawResults);
				json.put("treatGameNameAsFilepath", treatGameNameAsFilepath);

				final FileWriter fw = new FileWriter(file);
				bw = new BufferedWriter(fw);
				bw.write(json.toString(4));
				
			}
			catch (final Exception e)
			{
				e.printStackTrace();
			}
			finally
			{
				try
				{
					if (bw != null)
						bw.close();
				}
				catch (final Exception ex)
				{
					System.out.println("Error in closing the BufferedWriter" + ex);
				}
			}
		}
		
		public static TrialsBatchToRun fromJson(final String filepath)
		{
			try (final InputStream inputStream = new FileInputStream(new File(filepath)))
			{
				final JSONObject json = new JSONObject(new JSONTokener(inputStream));
				
				final String gameName = json.getString("gameName");
				final String ruleset = json.getString("ruleset");
				final int numTrials = json.getInt("numTrials");
				final int gameLengthCap = json.getInt("gameLengthCap");
				final double thinkingTime = json.getDouble("thinkingTime");
				final int iterationLimit = json.getInt("iterationLimit");
				final int warmingUpSecs = json.getInt("warmingUpSecs");
				final String outDir = json.getString("outDir");
				final JSONArray jArray = json.optJSONArray("agentStrings");
				final String[] agentStrings = jArray.toList().toArray(new String[0]);
				final boolean outputSummary = json.getBoolean("outputSummary");
				final boolean outputAlphaRankData = json.getBoolean("outputAlphaRankData");
				final boolean outputRawResults = json.getBoolean("outputRawResults");
				final boolean treatGameNameAsFilepath = json.optBoolean("treatGameNameAsFilepath", false);
				
				return new TrialsBatchToRun(gameName, ruleset, numTrials, gameLengthCap, thinkingTime, iterationLimit, 
						warmingUpSecs, outDir, agentStrings, outputSummary, outputAlphaRankData, outputRawResults, 
						treatGameNameAsFilepath);

			}
			catch (final Exception e)
			{
				e.printStackTrace();
				return null;
			}
		}
		
	}
	
	//-------------------------------------------------------------------------
	
	/**
	 * Can be used for quick testing without command-line args, or proper
	 * testing with elaborate setup through command-line args
	 * @param args
	 */
	@SuppressWarnings("unchecked")
	public static void main(final String[] args)
	{
		// Feature Set caching is safe in this main method
		JITSPatterNetFeatureSet.ALLOW_FEATURE_SET_CACHE = true;
		
		// Define options for arg parser
		final CommandLineArgParse argParse = 
				new CommandLineArgParse
				(
					true,
					"Evaluate many agents in many games sequentially. Configuration of all experiments to be run should be in a JSON file."
				);
		
		// Removed multithreading related options
		
		argParse.addOption(new ArgOption()
				.withNames("--json-files")
				.help("JSON files, each describing one batch of trials, which we should run in this job.")
				.withNumVals("+")
				.withType(OptionTypes.String)
				.setRequired());
		
		argParse.addOption(new ArgOption()
				.withNames("--useGUI")
				.help("Whether to create a small GUI that can be used to "
						+ "manually interrupt training run. False by default."));
		argParse.addOption(new ArgOption()
				.withNames("--max-wall-time")
				.help("Max wall time in minutes (or -1 for no limit).")
				.withDefault(Integer.valueOf(-1))
				.withNumVals(1)
				.withType(OptionTypes.Int));
		
		// Parse the args
		if (!argParse.parseArguments(args))
			return;

		// Use the parsed args
		final ComputeMultiAgents experiment = new ComputeMultiAgents();
		
		experiment.useGUI = argParse.getValueBool("--useGUI");
		experiment.maxWallTime = argParse.getValueInt("--max-wall-time");
		experiment.jsonFiles = (List<String>) argParse.getValue("--json-files");
		
		experiment.startExperiment();
	}

}
