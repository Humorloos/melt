package de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.processedNode;

import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.TextExtractorUrlFragment;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.constant.KBertLabelPropertyTypes;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.statement.LiteralObjectStatement;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.Statement;

import java.util.Comparator;
import java.util.Iterator;
import java.util.Optional;
import java.util.stream.StreamSupport;

public class ProcessedResource<T extends Resource> extends ProcessedRDFNode {
    protected final T resource;

    public ProcessedResource(T resource) {
        this.resource = resource;
    }

    public String getRaw() {
        Iterable<Statement> statements = resource::listProperties;
        return StreamSupport.stream(statements.spliterator(), false)
                .filter(s -> s.getObject().isLiteral())
                .map(LiteralObjectStatement::new)
                .filter(los -> los.getPredicate().getLabelType() != KBertLabelPropertyTypes.OTHER)
                .min(Comparator.comparing(s -> s.getPredicate().getLabelType()))
                .map(s -> s.getNeighbor().getRaw())
                .or(() -> {
                    Iterator<String> urlFragmentIterator = new TextExtractorUrlFragment().extract(resource).iterator();
                    if (urlFragmentIterator.hasNext()) return Optional.ofNullable(urlFragmentIterator.next());
                    else return Optional.ofNullable(resource.getURI());
                })
                .orElse(null);
    }
}
