'use strict';

module.exports = (sequelize, DataTypes) => {
  const DependentVariableResult = sequelize.define('DependentVariableResult', {
  	Name: DataTypes.STRING,
    Coefficient: DataTypes.FLOAT,
    Pval: DataTypes.FLOAT,
    Transformations: DataTypes.STRING,
    UnitRoot: DataTypes.STRING
  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'DependentVariableResult'
  });

  DependentVariableResult.associate = function(models) {
    models.DependentVariableResult.belongsTo(models.RunDetail, {
      onDelete: "CASCADE",
      foreignKey: "RunId"
    });
  };

  return DependentVariableResult;
};