import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.Map;

import org.apache.commons.rng.RandomProviderState;
import org.apache.commons.rng.core.RandomProviderDefaultState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import features.feature_sets.network.JITSPatterNetFeatureSet;
import java.util.regex.Pattern;
import game.Game;
import game.equipment.container.Container;
import game.match.Match;
import game.rules.end.End;
import game.rules.end.EndRule;
import game.rules.phase.Phase;
import game.rules.play.moves.Moves;
import game.types.board.SiteType;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import main.CommandLineArgParse;
import main.CommandLineArgParse.ArgOption;
import main.CommandLineArgParse.OptionTypes;
import main.DaemonThreadFactory;
import main.UnixPrintWriter;
import main.collections.ListUtils;
import main.Constants;
import manager.utils.game_logs.MatchRecord;
import metrics.Metric;
import metrics.MetricsTracker;
import metrics.Utils;
import metrics.single.boardCoverage.BoardCoverageDefault;
import metrics.single.boardCoverage.BoardCoverageFull;
import metrics.single.boardCoverage.BoardCoverageUsed;
import metrics.single.complexity.DecisionMoves;
import metrics.single.complexity.GameTreeComplexity;
import metrics.single.complexity.StateSpaceComplexity;
import metrics.single.duration.DurationActions;
import metrics.single.duration.DurationMoves;
import metrics.single.duration.DurationTurns;
import metrics.single.duration.DurationTurnsNotTimeouts;
import metrics.single.duration.DurationTurnsStdDev;
import metrics.single.outcome.AdvantageP1;
import metrics.single.outcome.Balance;
import metrics.single.outcome.Drawishness;
import metrics.single.outcome.OutcomeUniformity;
import metrics.single.outcome.Timeouts;
import metrics.single.outcome.Completion;
import other.GameLoader;
import other.concept.ConceptComputationType;
import other.concept.ConceptDataType;
import other.concept.ConceptType;
import other.playout.PlayoutMoveSelector;
import other.state.container.ContainerState;
import other.trial.Trial;
import other.context.Context;
import other.concept.Concept;
import other.move.Move;
import supplementary.experiments.concepts.ParallelComputeConceptsMultipleGames;
import supplementary.experiments.concepts.ParallelComputeConceptsMultipleGames.ConceptsJobOutput;



public class ComputeConcept extends ParallelComputeConceptsMultipleGames {
    public void processGame
    (
        Game game, String trialsDirString, String conceptsDirString, 
        int numThreads, int numTrials, int maxNumPlayoutActions, boolean isShort
    ) 
    {
        if (numThreads > numTrials) {
            numThreads = numTrials;
        }
        generateTrialsIfNeeded(game, trialsDirString, numTrials, numTrials, maxNumPlayoutActions);
        
        if (isShort) {
            computeConceptsShort(game, new File(trialsDirString), new File(conceptsDirString), numThreads);
        } else {
            computeConcepts(game, new File(trialsDirString), new File(conceptsDirString), numThreads);
        }
    }

    protected static void computeConceptsShort
	(
		final Game game, final File trialsDir, 
		final File conceptsDir, final int numThreads
	)
	{
		// Load all our trials
		final List<Trial> allTrials = new ArrayList<Trial>();
		final List<RandomProviderState> trialStartRNGs = new ArrayList<RandomProviderState>();
		
		for (final File trialFile : trialsDir.listFiles())
		{
			if (trialFile.getName().endsWith(".txt"))
			{
				MatchRecord loadedRecord;
				try
				{
					loadedRecord = MatchRecord.loadMatchRecordFromTextFile(trialFile, game);
					final Trial loadedTrial = loadedRecord.trial();
					allTrials.add(loadedTrial);
					trialStartRNGs.add(loadedRecord.rngState());
				}
				catch (final IOException e)
				{
					e.printStackTrace();
				}
			}
		}
		
		final List<Concept> ignoredConcepts = new ArrayList<Concept>();
		ignoredConcepts.add(Concept.Behaviour);
		ignoredConcepts.add(Concept.StateRepetition);
		ignoredConcepts.add(Concept.Duration);
		ignoredConcepts.add(Concept.Complexity);
		ignoredConcepts.add(Concept.BoardCoverage);
		ignoredConcepts.add(Concept.GameOutcome);
		ignoredConcepts.add(Concept.StateEvaluation);
		ignoredConcepts.add(Concept.Clarity);
		ignoredConcepts.add(Concept.Decisiveness);
		ignoredConcepts.add(Concept.Drama);
		ignoredConcepts.add(Concept.MoveEvaluation);
		ignoredConcepts.add(Concept.StateEvaluationDifference);
		ignoredConcepts.add(Concept.BoardSitesOccupied);
		ignoredConcepts.add(Concept.BranchingFactor);
		ignoredConcepts.add(Concept.DecisionFactor);
		ignoredConcepts.add(Concept.MoveDistance);
		ignoredConcepts.add(Concept.PieceNumber);
		ignoredConcepts.add(Concept.ScoreDifference);

        final List<Concept> conceptsToSave = Arrays.asList(
            Concept.DurationActions,
            Concept.DurationMoves,
            Concept.DurationTurns,
            Concept.DurationTurnsStdDev,
            Concept.DurationTurnsNotTimeouts,
            Concept.DecisionMoves,
            Concept.GameTreeComplexity,
            Concept.StateTreeComplexity,
            Concept.BoardCoverageDefault,
            Concept.BoardCoverageFull,
            Concept.BoardCoverageUsed,
            Concept.AdvantageP1,
            Concept.Balance,
            Concept.Completion,
            Concept.Drawishness,
            Concept.Timeouts,
            Concept.OutcomeUniformity
        );
		
		// Map from non-frequency-concept name to index of the frequency version of the concept
		final Map<String, Integer> conceptToFrequencyIndexMap = new HashMap<String, Integer>();
		for (int i = 0; i < Concept.values().length; ++i)
		{
			final Concept concept = Concept.values()[i];
			final int frequencyStringIndex = concept.name().indexOf("Frequency");
			
			if (frequencyStringIndex >= 0)
			{
				final String nonFrequencyConceptName = concept.name().substring(0, frequencyStringIndex);
				conceptToFrequencyIndexMap.put(nonFrequencyConceptName, Integer.valueOf(i));
			}
		}
		
		final BitSet gameBooleanConcepts = game.booleanConcepts();
		final Map<Integer, String> nonBooleanConceptsValues = game.nonBooleanConcepts();
		
		// Split trials into batches, each to be processed by a different thread
		final List<List<Trial>> trialsPerJob = ListUtils.split(allTrials, numThreads);
		final List<List<RandomProviderState>> rngStartStatesPerJob = ListUtils.split(trialStartRNGs, numThreads);
		
		@SuppressWarnings("resource")
		final ExecutorService threadPool = Executors.newFixedThreadPool(numThreads, DaemonThreadFactory.INSTANCE);

		try
		{
			final List<Future<ConceptsJobOutput>> conceptsJobOutputs = 
					new ArrayList<Future<ConceptsJobOutput>>();
			
			for (int jobIdx = 0; jobIdx < numThreads; ++jobIdx)
			{
				final List<Trial> trials = trialsPerJob.get(jobIdx);
				final List<RandomProviderState> rngStartStates = rngStartStatesPerJob.get(jobIdx);
				
				// Submit a job for this sublist of trials
				conceptsJobOutputs.add(threadPool.submit
				(
					() -> 
					{
						// We need a separate metrics tracker per job. Separate list of Metric
						// objects too, as we use them in a stateful manner
						final List<Metric> conceptMetrics = new ArrayList<Metric>();
						// Duration
						conceptMetrics.add(new DurationActions());
						conceptMetrics.add(new DurationMoves());
						conceptMetrics.add(new DurationTurns());
						conceptMetrics.add(new DurationTurnsStdDev());
						conceptMetrics.add(new DurationTurnsNotTimeouts());
						// Complexity
						conceptMetrics.add(new DecisionMoves());
						conceptMetrics.add(new GameTreeComplexity());
						conceptMetrics.add(new StateSpaceComplexity());
						// Board Coverage
						conceptMetrics.add(new BoardCoverageDefault());
						conceptMetrics.add(new BoardCoverageFull());
						conceptMetrics.add(new BoardCoverageUsed());
						// Outcome
						conceptMetrics.add(new AdvantageP1());
						conceptMetrics.add(new Balance());
						conceptMetrics.add(new Completion());
						conceptMetrics.add(new Drawishness());
						conceptMetrics.add(new Timeouts());
						conceptMetrics.add(new OutcomeUniformity());
						
						final MetricsTracker metricsTracker = new MetricsTracker(conceptMetrics);
						
						// Frequencies returned by all the playouts.
						final double[] frequencyPlayouts = new double[Concept.values().length];
						
						// Starting concepts
						final Map<String, Double> mapStarting = new HashMap<String, Double>();
						
						double numStartComponents = 0.0;
						double numStartComponentsHands = 0.0;
						double numStartComponentsBoard = 0.0;
						
						for (int trialIndex = 0; trialIndex < trials.size(); trialIndex++)
						{
							final Trial trial = trials.get(trialIndex);
							final RandomProviderState rngState = rngStartStates.get(trialIndex);

							// Setup a new instance of the game
							final Context context = Utils.setupNewContext(game, rngState);
							metricsTracker.startNewTrial(context, trial);
							
							// Compute the start concepts
							for (int cid = 0; cid < context.containers().length; cid++)
							{
								final Container cont = context.containers()[cid];
								final ContainerState cs = context.containerState(cid);
								if (cid == 0)
								{
									if (gameBooleanConcepts.get(Concept.Cell.id()))
									{
										for (int cell = 0; cell < cont.topology().cells().size(); cell++)
										{
											final int count = (game.hasSubgames() ? ((Match) game).instances()[0].getGame().isStacking() :  game.isStacking()) 
													? cs.sizeStack(cell, SiteType.Cell)
													: cs.count(cell, SiteType.Cell);
											numStartComponents += count;
											numStartComponentsBoard += count;
										}
									}

									if (gameBooleanConcepts.get(Concept.Vertex.id()))
									{
										for (int vertex = 0; vertex < cont.topology().vertices().size(); vertex++)
										{
											final int count = (game.hasSubgames() ? ((Match) game).instances()[0].getGame().isStacking() :  game.isStacking()) 
													? cs.sizeStack(vertex, SiteType.Vertex)
													: cs.count(vertex, SiteType.Vertex);
											numStartComponents += count;
											numStartComponentsBoard += count;
										}
									}

									if (gameBooleanConcepts.get(Concept.Edge.id()))
									{
										for (int edge = 0; edge < cont.topology().edges().size(); edge++)
										{
											final int count = (game.hasSubgames() ? ((Match) game).instances()[0].getGame().isStacking() :  game.isStacking())  
													? cs.sizeStack(edge, SiteType.Edge)
													: cs.count(edge, SiteType.Edge);
											numStartComponents += count;
											numStartComponentsBoard += count;
										}
									}
								}
								else
								{
									if (gameBooleanConcepts.get(Concept.Cell.id()))
									{
										for (int cell = context.sitesFrom()[cid]; cell < context.sitesFrom()[cid]
												+ cont.topology().cells().size(); cell++)
										{
											final int count = (game.hasSubgames() ? ((Match) game).instances()[0].getGame().isStacking() :  game.isStacking()) 
													? cs.sizeStack(cell, SiteType.Cell)
													: cs.count(cell, SiteType.Cell);
											numStartComponents += count;
											numStartComponentsHands += count;
										}
									}
								}
							}

							// Frequencies returned by this playout.
							final double[] frequencyPlayout = new double[Concept.values().length];

							// Run the playout.
							int turnsWithMoves = 0;
							Context prevContext = null;
							for (int i = trial.numInitialPlacementMoves(); i < trial.numMoves(); i++)
							{
								final Moves legalMoves = context.game().moves(context);

								final int numLegalMoves = legalMoves.moves().size();
								if (numLegalMoves > 0)
									turnsWithMoves++;

								for (final Move legalMove : legalMoves.moves())
								{
									final BitSet moveConcepts = legalMove.moveConcepts(context);
									
									for (int indexConcept = 0; indexConcept < Concept.values().length; indexConcept++)
									{
										final Concept concept = Concept.values()[indexConcept];
										if (moveConcepts.get(concept.id()))
										{
											if (conceptToFrequencyIndexMap.containsKey(concept.name()))
											{
												final int frequencyConceptIdx = conceptToFrequencyIndexMap.get(concept.name()).intValue();
												frequencyPlayout[frequencyConceptIdx] += 1.0 / numLegalMoves;
											}
											else
											{
												frequencyPlayout[concept.id()] += 1.0 / numLegalMoves;
											}
										}
									}
								}

								// We keep the context before the ending state for the frequencies of the end
								// conditions.
								if (i == trial.numMoves() - 1)
									prevContext = new Context(context);

								// We go to the next move.
								context.game().apply(context, trial.getMove(i));
								metricsTracker.observeNextState(context);
							}
							
							metricsTracker.observeFinalState(context);
							
							// Compute avg for all the playouts.
							for (int j = 0; j < frequencyPlayout.length; j++)
								frequencyPlayouts[j] += frequencyPlayout[j] / turnsWithMoves;

							context.trial().lastMove().apply(prevContext, true);

							boolean noEndFound = true;

							if (context.rules().phases() != null)
							{
								final int mover = context.state().mover();
								final Phase endPhase = context.rules().phases()[context.state().currentPhase(mover)];
								final End endPhaseRule = endPhase.end();

								// Only check if action not part of setup
								if (context.active() && endPhaseRule != null)
								{
									final EndRule[] endRules = endPhaseRule.endRules();
									for (final EndRule endingRule : endRules)
									{
										final EndRule endRuleResult = endingRule.eval(prevContext);
										if (endRuleResult == null)
											continue;

										final BitSet endConcepts = endingRule.stateConcepts(prevContext);

										noEndFound = false;
										for (int indexConcept = 0; indexConcept < Concept.values().length; indexConcept++)
										{
											final Concept concept = Concept.values()[indexConcept];
											if (concept.type().equals(ConceptType.End) && endConcepts.get(concept.id()))
											{
												if (conceptToFrequencyIndexMap.containsKey(concept.name()))
												{
													final int frequencyConceptIdx = conceptToFrequencyIndexMap.get(concept.name()).intValue();
													frequencyPlayouts[frequencyConceptIdx]++; 
												}
												else
												{
													frequencyPlayout[concept.id()]++;
												}
											}
										}
										break;
									}
								}
							}

							final End endRule = context.rules().end();
							if (noEndFound && endRule != null)
							{
								final EndRule[] endRules = endRule.endRules();
								for (final EndRule endingRule : endRules)
								{
									final EndRule endRuleResult = endingRule.eval(prevContext);
									if (endRuleResult == null)
										continue;

									final BitSet endConcepts = endingRule.stateConcepts(prevContext);

									noEndFound = false;
									for (int indexConcept = 0; indexConcept < Concept.values().length; indexConcept++)
									{
										final Concept concept = Concept.values()[indexConcept];
										if (concept.type().equals(ConceptType.End) && endConcepts.get(concept.id()))
										{
											if (conceptToFrequencyIndexMap.containsKey(concept.name()))
											{
												final int frequencyConceptIdx = conceptToFrequencyIndexMap.get(concept.name()).intValue();
												frequencyPlayouts[frequencyConceptIdx]++; 
											}
											else
											{
												frequencyPlayout[concept.id()]++;
											}
										}
									}
									break;
								}
							}

							if (noEndFound)
							{
								frequencyPlayouts[Concept.DrawFrequency.ordinal()]++; 
							}
						}

						final TDoubleArrayList frequenciesThisJob = TDoubleArrayList.wrap(new double[Concept.values().length]);
						for (int indexConcept = 0; indexConcept < Concept.values().length; indexConcept++)
						{
							frequenciesThisJob.setQuick(indexConcept, frequencyPlayouts[indexConcept] / trials.size());
						}
						
						final Map<String, Double> metricsThisJob = metricsTracker.finaliseMetrics(game, trials.size());
						
						// Finish computing the starting concepts
						mapStarting.put(Concept.NumStartComponents.name(), Double.valueOf(numStartComponents / trials.size()));
						mapStarting.put(Concept.NumStartComponentsHand.name(), Double.valueOf(numStartComponentsHands / trials.size()));
						mapStarting.put(Concept.NumStartComponentsBoard.name(), Double.valueOf(numStartComponentsBoard / trials.size()));

						mapStarting.put(Concept.NumStartComponentsPerPlayer.name(), Double.valueOf((numStartComponents / trials.size()) / (game.players().count() == 0 ? 1 : game.players().count())));
						mapStarting.put(Concept.NumStartComponentsHandPerPlayer.name(), Double.valueOf((numStartComponentsHands / trials.size()) / (game.players().count() == 0 ? 1 : game.players().count())));
						mapStarting.put(Concept.NumStartComponentsBoardPerPlayer.name(), Double.valueOf((numStartComponentsBoard / trials.size()) / (game.players().count() == 0 ? 1 : game.players().count())));
						
						return new ConceptsJobOutput(frequenciesThisJob, metricsThisJob, mapStarting);
					}
				));
			}
			
			// Final wrap-up which is not really nicely parallelisable
			final Map<String, Double> conceptValues = ConceptsJobOutput.mergeResults(conceptsJobOutputs, trialsPerJob);
			
			// Computation of the p/s and m/s
			final Trial trial = new Trial(game);
			final Context context = new Context(game, trial);

			// Warming up
			long stopAt = 0L;
			long start = System.nanoTime();
			final double warmingUpSecs = 10;
			final double measureSecs = 30;
			double abortAt = start + warmingUpSecs * 1000000000.0;
			while (stopAt < abortAt)
			{
				game.start(context);
				game.playout(context, null, 1.0, null, -1, Constants.UNDEFINED, ThreadLocalRandom.current());
				stopAt = System.nanoTime();
			}
			System.gc();

			// Set up RNG for this game, Always with a rng of 2077.
			final Random rng = new Random((long) game.name().hashCode() * 2077);

			// The Test
			stopAt = 0L;
			start = System.nanoTime();
			abortAt = start + measureSecs * 1000000000.0;
			int playouts = 0;
			int moveDone = 0;
			while (stopAt < abortAt)
			{
				game.start(context);
				game.playout(context, null, 1.0, null, -1, Constants.UNDEFINED, rng);
				moveDone += context.trial().numMoves();
				stopAt = System.nanoTime();
				++playouts;
			}

			final double secs = (stopAt - start) / 1000000000.0;
			final double rate = (playouts / secs);
			final double rateMove = (moveDone / secs);
			conceptValues.put(Concept.PlayoutsPerSecond.name(), Double.valueOf(rate));
			conceptValues.put(Concept.MovesPerSecond.name(), Double.valueOf(rateMove));
			
			// Write file with concepts for this game/ruleset
			String conceptsFilepath = conceptsDir.getAbsolutePath();
			conceptsFilepath = conceptsFilepath.replaceAll(Pattern.quote("\\"), "/");
			if (!conceptsFilepath.endsWith("/"))
				conceptsFilepath += "/";
			conceptsFilepath += "Concepts.csv";
			
			final File conceptsFile = new File(conceptsFilepath);
			conceptsFile.getParentFile().mkdirs();
			try (final PrintWriter writer = new UnixPrintWriter(conceptsFile, "UTF-8"))
			{
				final DecimalFormat doubleFormatter = new DecimalFormat("##.##");
				
				final StringBuilder header = new StringBuilder();
                for (final Concept concept : conceptsToSave)
                {
                    header.append(concept.name()).append(",");
                }
                header.deleteCharAt(header.length() - 1);		// 最後のカンマを削除
                writer.println(header);
				
				// Now write the concept values
                final StringBuilder conceptValuesLine = new StringBuilder();
                for (int i = 0; i < conceptsToSave.size(); ++i)
                {
                    final Concept concept = conceptsToSave.get(i);
                    final String conceptName = concept.name();
                    
                    if (concept.dataType() == ConceptDataType.BooleanData)
                    {
                        // Boolean concept
                        if (ignoredConcepts.contains(concept))
                            conceptValuesLine.append("NULL");
                        else if (gameBooleanConcepts.get(concept.id()))
                            conceptValuesLine.append(1);
                        else
                            conceptValuesLine.append(0);
                    }
                    else
                    {
                        if (concept.computationType() == ConceptComputationType.Compilation)
                        {
                            // Non-boolean compilation concept
                            conceptValuesLine.append(nonBooleanConceptsValues.get(Integer.valueOf(concept.id())));
                        }
                        else
                        {
                            // Concept computed from playouts
                            if (conceptValues.get(conceptName) == null)
                            {
                                conceptValuesLine.append("NULL");
                            }
                            else
                            {
                                final double value = conceptValues.get(conceptName).doubleValue();
                                conceptValuesLine.append(doubleFormatter.format(value));
                            }
                        }
                    }
                    
                    if (i + 1 < conceptsToSave.size())
                        conceptValuesLine.append(",");
                }
                writer.println(conceptValuesLine);
			}
			catch (final Exception e)
			{
				e.printStackTrace();
			}
		}
		catch (final Exception e)
		{
			e.printStackTrace();
		}
		finally
		{
			threadPool.shutdown();
			try 
			{
				threadPool.awaitTermination(24, TimeUnit.HOURS);
			} 
			catch (final InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

    private int countExistingTrialFiles(File trialsDir)
    {
        if (!trialsDir.exists() || !trialsDir.isDirectory())
        {
            return 0;
        }

        int count = 0;
        try
        {
            File[] files = trialsDir.listFiles();
            if (files != null)
            {
                for (File file : files)
                {
                    if (file.isFile() && file.getName().endsWith(".txt"))
                    {
                        count++;
                    }
                }
            }
        }
        catch (SecurityException e)
        {
            System.err.println("Error accessing trials directory: " + e.getMessage());
            e.printStackTrace();
        }
        return count;
    }

    private void generateTrialsIfNeeded(Game game, String trialsDirString, int numThreads, int numTrials, int maxNumPlayoutActions)
    {
        File trialsDir = new File(trialsDirString);
        int numExistingTrialFiles = countExistingTrialFiles(trialsDir);
        
        if (numExistingTrialFiles < numTrials)
        {
            int numTrialsToRun = numTrials - numExistingTrialFiles;
            generateRandomTrialsWithMaxActions(game, numTrialsToRun, numExistingTrialFiles, trialsDir, numThreads, null, game.getOptions(), maxNumPlayoutActions);
        }
    }

    protected static void generateRandomTrialsWithMaxActions(Game game, int numTrialsToRun, int firstTrialIndex, File trialsDir, int numThreads, String gameName, List<String> gameOptions, int maxNumPlayoutActions) {
        ExecutorService threadPool = Executors.newFixedThreadPool(numThreads, DaemonThreadFactory.INSTANCE);
        TIntArrayList trialIndicesToGenerate = ListUtils.range(firstTrialIndex, firstTrialIndex + numTrialsToRun);
        TIntArrayList[] trialIndicesPerJob = ListUtils.split(trialIndicesToGenerate, numThreads);

        try {
            TIntArrayList[] var10 = trialIndicesPerJob;
            int var11 = trialIndicesPerJob.length;

            for(int var12 = 0; var12 < var11; ++var12) {
                TIntArrayList trialIndices = var10[var12];
                threadPool.submit(() -> {
                for(int i = 0; i < trialIndices.size(); ++i) {
                    int trialIdx = trialIndices.getQuick(i);

                    try {
                        Trial trial = new Trial(game);
                        Context context = new Context(game, trial);
                        RandomProviderDefaultState gameStartRngState = (RandomProviderDefaultState)context.rng().saveState();
                        game.start(context);
                        game.playout(context, null, 1.0, (PlayoutMoveSelector)null, 0, maxNumPlayoutActions, ThreadLocalRandom.current());
                        String trialFilepath = trialsDir.getAbsolutePath();
                        trialFilepath = trialFilepath.replaceAll(Pattern.quote("\\"), "/");
                        if (!trialFilepath.endsWith("/")) {
                            trialFilepath = trialFilepath + "/";
                        }

                        trialFilepath = trialFilepath + "Trial_" + trialIdx + ".txt";
                        trial.saveTrialToTextFile(new File(trialFilepath), gameName, gameOptions, gameStartRngState);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                });
            }
        } catch (Exception var22) {
            var22.printStackTrace();
        } finally {
            threadPool.shutdown();
            try {
                threadPool.awaitTermination(24L, TimeUnit.HOURS);
            } catch (InterruptedException var21) {
                var21.printStackTrace();
            }
        }
    }
    

    public static void main(final String[] args)
	{
        // Feature Set caching is safe in this main method
		JITSPatterNetFeatureSet.ALLOW_FEATURE_SET_CACHE = true;
		
		// Define options for arg parser
		final CommandLineArgParse argParse = 
				new CommandLineArgParse
				(
					true,
					"Compute concepts for multiple games."
				);
		
		argParse.addOption(new ArgOption()
				.withNames("--trials-dir")
				.help("Trial files, each containing a single trial for which we should compute concepts in this job.")
                .withNumVals(1)
                .withType(OptionTypes.String)
				.setRequired());
		argParse.addOption(new ArgOption()
				.withNames("--concepts-dir")
				.help("Concept files, each containing the concepts for a single trial.")
				.withNumVals(1)
				.withType(OptionTypes.String)
				.setRequired());
		argParse.addOption(new ArgOption()
				.withNames("--game-path")
				.help("Path to the game file.")
				.withNumVals(1)
				.withType(OptionTypes.String)
				.setRequired());
        argParse.addOption(new ArgOption()
				.withNames("--num-threads")
				.help("Number of threads to use.")
				.withNumVals(1)
                .withDefault(Integer.valueOf(1))
				.withType(OptionTypes.Int));
        argParse.addOption(new ArgOption()
                .withNames("--num-trials")
                .help("Number of trials to generate.")
                .withNumVals(1)
                .withDefault(Integer.valueOf(100))
                .withType(OptionTypes.Int));
        argParse.addOption(new ArgOption()
                .withNames("--max-num-playout-actions")
                .help("Maximum number of playout actions to generate.")
                .withNumVals(1)
                .withDefault(Integer.valueOf(2500))
                .withType(OptionTypes.Int));
        argParse.addOption(new ArgOption()
                .withNames("--short")
                .help("Whether to use the short version of the game.")
                .withNumVals(1)
                .withDefault(Boolean.valueOf(false))
                .withType(OptionTypes.Boolean));
		
		// Parse the args
		if (!argParse.parseArguments(args))
			return;
        
        // Get the parsed args
        String trialsDirString = argParse.getValueString("--trials-dir");
        String conceptsDirString = argParse.getValueString("--concepts-dir");
        String gamePathString = argParse.getValueString("--game-path");
        int numThreads = argParse.getValueInt("--num-threads");
        int numTrials = argParse.getValueInt("--num-trials");
        int maxNumPlayoutActions = argParse.getValueInt("--max-num-playout-actions");
        boolean isShort = argParse.getValueBool("--short");

        Game game = GameLoader.loadGameFromFile(new File(gamePathString), "");
		final ComputeConcept experiment = new ComputeConcept();
        experiment.processGame(game, trialsDirString, conceptsDirString, numThreads, numTrials, maxNumPlayoutActions, isShort);
    }
}
