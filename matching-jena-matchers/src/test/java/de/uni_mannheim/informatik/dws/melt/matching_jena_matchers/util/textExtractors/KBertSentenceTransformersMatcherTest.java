package de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors;

import de.uni_mannheim.informatik.dws.melt.matching_data.TestCase;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.TextMoleculeExtractorImpl;
import de.uni_mannheim.informatik.dws.melt.matching_ml.python.nlptransformers.kbert.KBertSentenceTransformersMatcher;
import org.apache.commons.io.FileUtils;
import org.apache.jena.ontology.OntModel;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URI;
import java.net.URL;
import java.util.Arrays;
import java.util.Map;
import java.util.Properties;

import static de.uni_mannheim.informatik.dws.melt.matching_base.typetransformer.TypeTransformerRegistry.getTransformedObject;
import static de.uni_mannheim.informatik.dws.melt.matching_base.typetransformer.TypeTransformerRegistry.getTransformedPropertiesOrNewInstance;
import static de.uni_mannheim.informatik.dws.melt.matching_ml.python.PythonServer.PYTHON_DIRECTORY_NAME;
import static java.nio.file.Files.createDirectories;

public class KBertSentenceTransformersMatcherTest {

    @Test
    public void testGenerateKBertInputVariations() throws Exception {
        TestCase testCase = TrackRepository.Anatomy.Default.getTestCase(0);
        File rootFile = new File(
                new File(
                        this.getClass().getProtectionDomain().getCodeSource().getLocation().getFile()
                ).getParentFile().getParentFile().getParentFile(),
                "matching-ml-python/" + PYTHON_DIRECTORY_NAME + "/kbert/test/resources/TM/" +
                        testCase.getTrack().getName() + '/' + testCase.getName()
        );
        URL parameters = testCase.getParameters().toURL();
        Properties properties = getTransformedPropertiesOrNewInstance(parameters);
        for (Boolean normalized : Arrays.asList(true, false)) {
            for (Boolean allTargets : Arrays.asList(true, false)) {
                for (Boolean multiText : Arrays.asList(true, false)) {
                    for (Map.Entry<String, URI> entry : Map.of("corpus", testCase.getSource(), "queries", testCase.getTarget()).entrySet()) {
                        URL source = entry.getValue().toURL();
                        OntModel sourceOntology = getTransformedObject(source, OntModel.class, properties);
                        KBertSentenceTransformersMatcher matcher = new KBertSentenceTransformersMatcher(
                                new TextMoleculeExtractorImpl(allTargets, normalized, multiText), "paraphrase-MiniLM-L6-v2");
                        File targetFile = new File(rootFile,
                                KBertSentenceTransformersMatcher.NORMALIZED_MAP.get(normalized) + "/" +
                                        KBertSentenceTransformersMatcher.ALL_TARGETS_MAP.get(allTargets) + "/" +
                                        "isMulti_" + multiText + "/" +
                                        entry.getKey() + ".csv");

                        if (targetFile.exists()) FileUtils.delete(targetFile);
                        createDirectories(targetFile.getParentFile().toPath());
                        // when
                        matcher.createTextFile(sourceOntology, targetFile, matcher.getResourcesExtractor().get(0), properties);
                    }
                }
            }
        }
        // given


        // then
        System.out.println("");
    }
}
