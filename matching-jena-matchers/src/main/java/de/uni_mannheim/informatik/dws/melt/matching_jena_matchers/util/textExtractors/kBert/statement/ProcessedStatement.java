package de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.statement;

import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.processedNode.ProcessedProperty;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.processedNode.ProcessedRDFNode;
import org.apache.jena.rdf.model.Statement;

import java.util.Map;
import java.util.Objects;

public abstract class ProcessedStatement<T extends ProcessedRDFNode> {
    protected NeighborRole role;

    protected final ProcessedProperty predicate;
    protected T neighbor;

    public void setUseIndex(boolean useIndex) {
        this.useIndex = useIndex;
    }

    private boolean useIndex = true;

    public ProcessedStatement(Statement statement) {
        this.predicate = new ProcessedProperty(statement.getPredicate());
        this.neighbor = null;
        this.role = null;
    }

    public Map<String, String> getRow() {
        String predicateRepr;
        String neighborRepr;
        if (useIndex) {
            predicateRepr = predicate.getKey();
            neighborRepr = neighbor.getKey();
        } else {
            predicateRepr = predicate.getNormalized();
            neighborRepr = neighbor.getNormalized();
        }
        return Map.of("p", predicateRepr, "n", neighborRepr, "r", role.getRole());
    }

    public T getNeighbor() {
        return neighbor;
    }

    public ProcessedProperty getPredicate() {
        return predicate;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final ProcessedStatement<T> other = (ProcessedStatement<T>) obj;
        return Objects.equals(getNeighbor(), other.getNeighbor()) &&
                Objects.equals(getPredicate(), other.getPredicate());
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 59 * hash + Objects.hashCode(neighbor.getNormalized() + predicate.getNormalized());
        return hash;
    }
}
