'use strict';

module.exports = (sequelize, DataTypes) => {
  const RunDetail = sequelize.define('RunDetail', {
      StartDate: DataTypes.DATE,
      EndDate: DataTypes.DATE,
      Status: DataTypes.STRING,
      ModelType: DataTypes.STRING,
      EndDateTimeForTests: DataTypes.DATE
  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'RunDetail'
  });

  RunDetail.associate = function(models) {
    models.RunDetail.hasMany(models.UserInput, {
      foreignKey: 'RunId'
    });
    models.RunDetail.hasMany(models.ModelRunDetail, {
      foreignKey: 'RunId'
    });
    models.RunDetail.hasMany(models.DummyVariable, {
      foreignKey: 'RunId'
    });
    models.RunDetail.hasMany(models.DependentVariableResult, {
      foreignKey: 'RunId'
    });
    models.RunDetail.hasMany(models.Shortlist, {
      foreignKey: 'RunId'
    });
    models.RunDetail.hasMany(models.IndependentVariableResult, {
      foreignKey: 'RunId'
    });
    models.RunDetail.hasMany(models.IntermediateOutput, {
      foreignKey: 'RunId'
    });
  };
  return RunDetail;
};