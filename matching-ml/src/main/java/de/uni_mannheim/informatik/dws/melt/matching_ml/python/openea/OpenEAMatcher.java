
package de.uni_mannheim.informatik.dws.melt.matching_ml.python.openea;

import de.uni_mannheim.informatik.dws.melt.matching_jena.MatcherYAAAJena;
import de.uni_mannheim.informatik.dws.melt.matching_ml.python.PythonServer;
import de.uni_mannheim.informatik.dws.melt.matching_ml.util.VectorOperations;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Alignment;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.AlignmentParser;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Correspondence;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.CorrespondenceRelation;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.regex.Pattern;
import org.apache.commons.io.FileUtils;
import org.apache.jena.ontology.OntModel;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This matching module uses the <a href="https://github.com/nju-websoft/OpenEA">OpenEA</a> library to match entities.
 * It uses all correspondences which are provided through either the constructor or match method(inputalignment)
 * with equivalence relation to train the approach. It only need positive correspondences and samples negative correspondences on its own.
 * <br>
 * If you apply your own configuration you can use the parameters from <a href="https://github.com/nju-websoft/OpenEA/blob/master/run/main_with_args.py#L30">openEA</a> and also
 * the following additional parameters:
 * <ul>
 * <li>predict_top_k - the number of matching entities which should at least retrived for one single entity</li>
 * <li>predict_min_sim_value - the similarity measure which should be applied for every correspondence. All sim values are greater than the given one (not equal or greater).</li>
 * </ul>
 * 
 */
public class OpenEAMatcher extends MatcherYAAAJena{

    private static final Logger LOGGER = LoggerFactory.getLogger(OpenEAMatcher.class);
    
    private static final File DEFAULT_BASE_DIRECTORY = new File(System.getProperty("java.io.tmpdir")); //OR: new File("./")
    
    /**
     * Map of URIs (String) to double array.
     * This represents the entities embeddings generated by openea for the left entities.
     */
    public static final String KB1_ENTITY_EMBEDDING = "http://oaei.ontologymatching.org/kb1_entity_embedding";
    
    /**
     * Map of URIs (String) to double array.
     * This represents the entities embeddings generated by openea for the right entities.
     */
    public static final String KB2_ENTITY_EMBEDDING = "http://oaei.ontologymatching.org/kb2_entity_embedding";
    
    /**
     * The base directory where all files are written.
     * In this folder, another folder with a random number is created to allow parallel execution of this matcher.
     */
    private File baseDirectory;
    
    /**
     * The fraction how many correspondences belongt to train (rest to validation).
     */
    private double fractionTrainCorrespondences;
    
    private long randomSeed;
    
    private OpenEAConfiguration config;
    private boolean loadEmbeddings;
    
    /**
     * Constructor which sets all variables.
     * @param baseDirectory the base directory where all files are written. In this folder, another folder with a random number is created to allow parallel execution of this matcher.
     * @param config the configuration for the model
     * @param loadEmbeddings true if also the embeddings should be loaded.
     * @param fractionTrainCorrespondences the fraction of training correspondences
     * @param randomSeed the random seed for splitting the alignment into train and validation alignment.
     */
    public OpenEAMatcher(File baseDirectory, OpenEAConfiguration config, boolean loadEmbeddings, double fractionTrainCorrespondences, long randomSeed) {
        this.baseDirectory = baseDirectory;
        this.config = config;
        this.loadEmbeddings = loadEmbeddings;
        this.fractionTrainCorrespondences = fractionTrainCorrespondences;
        this.randomSeed = randomSeed;
    }
    
    /**
     * Constructor with the config for training.
     * @param config the configuration for the model
     * @param fractionTrainCorrespondences the fraction of training correspondences
     * @param randomSeed the random seed for splitting the alignment into train and validation alignment.
     */
    public OpenEAMatcher(OpenEAConfiguration config, double fractionTrainCorrespondences, long randomSeed) {
        this(DEFAULT_BASE_DIRECTORY, config, false, fractionTrainCorrespondences, randomSeed);
    }
    
    /**
     * Makes a 80/20 train validation split and uses the default configuration and base folder.
     */
    public OpenEAMatcher() {
        this(DEFAULT_BASE_DIRECTORY, getDefaultConfig(), false, 0.8, 1234);
    }
    
    private static OpenEAConfiguration getDefaultConfig(){
        return new OpenEAConfiguration(OpenEAMatcher.class.getResourceAsStream("/openea_default_arguments.json"));
    }
    
    @Override
    public Alignment match(OntModel source, OntModel target, Alignment inputAlignment, Properties parameter) throws Exception {
        if(inputAlignment.isEmpty()){
            LOGGER.error("Given alignment is empty - no training correspondences. Abort");
            return inputAlignment;
        }
        File matchFolder = getRandomSubFolderOfBase();
        matchFolder.mkdirs();
        File rel1 = new File(matchFolder, "rel_triples_1");
        File attr1 = new File(matchFolder, "attr_triples_1");
        File rel2 = new File(matchFolder, "rel_triples_2");
        File attr2 = new File(matchFolder, "attr_triples_2");
        
        File alignmentFolder = new File(matchFolder, "alignment");
        alignmentFolder.mkdirs();

        File outputFolder = new File(matchFolder, "output");
        outputFolder.mkdirs();
        
        File configFile = new File(matchFolder, "arguments.json");
        this.config.addFileLocations(
                matchFolder.getAbsolutePath() + File.separator,
                outputFolder.getAbsolutePath() + File.separator, 
                "alignment" + File.separator
        );
        if(this.config.containsKey("predict_top_k") == false && this.config.containsKey("predict_min_sim_value") == false){
            LOGGER.info("Automatically set predict_top_k to one. Because otherwise no predictions would be generated.");
            this.config.addArgument("predict_top_k", 1);
        }
        
        try{
            LOGGER.debug("Write OpenEA config file");
            this.config.writeArgumentsToFile(configFile);
            LOGGER.debug("Write source graph to OpenEA specific files");
            writeKnowledgeGraphToFile(source, rel1, attr1);
            LOGGER.debug("Write target graph to OpenEA specific files");
            writeKnowledgeGraphToFile(target, rel2, attr2);
            LOGGER.debug("Write alignment files for OpenEA");
            writeTrainValAlignments(inputAlignment, alignmentFolder);
            
            PythonServer.getInstance().runOpenEAModel(configFile, this.loadEmbeddings);
            if(this.loadEmbeddings){
                if(parameter == null)
                    parameter = new Properties();
                parameter.put(KB1_ENTITY_EMBEDDING, VectorOperations.readVectorFile(new File(outputFolder, "kg1_ent_embeds_txt")));
                parameter.put(KB2_ENTITY_EMBEDDING, VectorOperations.readVectorFile(new File(outputFolder, "kg2_ent_embeds_txt")));
            }
            File predictionFile = new File(outputFolder, "topk.tsv");
            if(predictionFile.exists() == false){
                LOGGER.warn("The output file does not exist. Something went wrong during execution of OpenEA matcher. The input alignment is returned.");
                return inputAlignment;
            }
            return AlignmentParser.parseCSVWithoutHeader(predictionFile, '\t');
        } catch (IOException ex) {
            LOGGER.error("Could not read/write files for OpenEA library. Returning input alignment.", ex);
            return inputAlignment;
        } catch (Exception ex) {
            LOGGER.error("Some error in OpenEA server call occured. Returning input alignment.", ex);
            return inputAlignment;
        }finally{
            try {
                FileUtils.deleteDirectory(matchFolder);
            } catch (IOException ex) {
                LOGGER.warn("Could not delete the directory {} which contains temporary files for OpenEA matcher. Please delete them on your own.", matchFolder, ex);
            }
        }
    }
    

    protected void writeTrainValAlignments(Alignment a, File alignmentFolder) throws IOException{
        ArrayList<Correspondence> correspondenceList = new ArrayList<>();
        for(Correspondence c : a.getCorrespondencesRelation(CorrespondenceRelation.EQUIVALENCE)){
            correspondenceList.add(c);
        }
        Collections.shuffle(correspondenceList, new Random(this.randomSeed));
        
        int trainPos = (int)Math.round((double)correspondenceList.size() * this.fractionTrainCorrespondences);
        
        List<Correspondence> trainCorrespondences = correspondenceList.subList(0, trainPos);
        List<Correspondence> valCorrespondences = correspondenceList.subList(trainPos, correspondenceList.size());
                
        LOGGER.debug("Write train/val alignment files with {}/{} correspondences", 
                trainCorrespondences.size(), valCorrespondences.size());

        writeCorrespondences(trainCorrespondences, new File(alignmentFolder, "train_links"));
        writeCorrespondences(valCorrespondences, new File(alignmentFolder, "valid_links"));
        writeCorrespondences(new ArrayList<>(), new File(alignmentFolder, "test_links")); //file test_links has to exist
        
        //https://stackoverflow.com/questions/13483430/how-to-make-rounded-percentages-add-up-to-100
        //java rounding split percentage
    }
    
    
    protected void writeCorrespondences(List<Correspondence> correspondences, File file) throws IOException{
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"))) {
            for(Correspondence c : correspondences){
                writer.write(c.getEntityOne() + "\t" + c.getEntityTwo());
                writer.newLine();
            }
        }
    }
    
    private static Pattern CONTROL_CODE_PATTERN = Pattern.compile("[\t\r\n\f]");
    protected void writeKnowledgeGraphToFile(Model m, File relTriples, File attrTriples) throws IOException{       
        try (BufferedWriter relWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(relTriples), "UTF-8"));
             BufferedWriter attrWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(attrTriples), "UTF-8"));) {
            StmtIterator i = m.listStatements();
            while (i.hasNext()) {
                Statement s = i.next();
                if (s.getObject().isLiteral()) {
                    attrWriter.write(
                            s.getSubject().toString() + "\t" + 
                            s.getPredicate().toString() + "\t" +
                            CONTROL_CODE_PATTERN.matcher(s.getObject().asLiteral().getLexicalForm()).replaceAll(" ")
                    );
                    attrWriter.newLine();
                }else{
                    relWriter.write(
                            s.getSubject().toString() + "\t" + 
                            s.getPredicate().toString() + "\t" +
                            s.getObject().toString()
                    );
                    relWriter.newLine();
                }
            }
        }
    }
    
    private static final SecureRandom random = new SecureRandom();
    protected File getRandomSubFolderOfBase(){        
        long n = random.nextLong();
        n = (n == Long.MIN_VALUE) ? 0 : Math.abs(n);
        return new File(this.baseDirectory, "openeaDataset-" + Long.toString(n));
    }
}
